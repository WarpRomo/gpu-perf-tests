#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#define PORT 12345
#define IB_PORT 1
#define GID_IDX 1 // usually 1 for Soft RoCE on IPV4

// Macros
#define CUDA_CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Err: " << cudaGetErrorString(err) << std::endl; exit(1); } }
#define DIE(msg) { std::cerr << "Error: " << msg << " (" << strerror(errno) << ")" << std::endl; exit(1); }

struct ExchangeData {
    uint32_t lid;
    uint32_t qp_num;
    union ibv_gid gid;
};

// --- TCP HELPER FUNCTIONS ---
int setup_tcp_server() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    if(bind(sockfd, (struct sockaddr*)&address, sizeof(address)) < 0) DIE("Bind failed");
    listen(sockfd, 1);
    std::cout << "[TCP] Waiting for client..." << std::endl;
    int new_sock = accept(sockfd, NULL, NULL);
    return new_sock;
}

int setup_tcp_client(const char* ip) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, ip, &serv_addr.sin_addr);
    
    std::cout << "[TCP] Connecting to " << ip << "..." << std::endl;
    while (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cout << "Retrying..." << std::endl;
        sleep(1);
    }
    return sock;
}

void tcp_send_data(int sock, void* buf, size_t size) {
    size_t total = 0;
    while(total < size) {
        ssize_t sent = send(sock, (char*)buf + total, size - total, 0);
        if (sent < 0) DIE("TCP Send");
        total += sent;
    }
}

void tcp_recv_data(int sock, void* buf, size_t size) {
    size_t total = 0;
    while(total < size) {
        ssize_t r = recv(sock, (char*)buf + total, size - total, MSG_WAITALL);
        if (r <= 0) DIE("TCP Recv");
        total += r;
    }
}

// --- RDMA CONTEXT ---
struct RDMAContext {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    struct ibv_port_attr port_attr;
    int lid;
    union ibv_gid gid;
    
    // Remote info
    uint32_t remote_qpn;
    uint32_t remote_lid;
    union ibv_gid remote_gid;

    void init(void* buf, size_t size) {
        struct ibv_device **dev_list = ibv_get_device_list(NULL);
        if (!dev_list) DIE("No IB devices found");
        
        // Find rxe0
        struct ibv_device *ib_dev = dev_list[0]; // Assuming rxe0 is first or only
        const char* dev_name = ibv_get_device_name(ib_dev);
        std::cout << "[RDMA] Using device: " << dev_name << std::endl;

        ctx = ibv_open_device(ib_dev);
        if (!ctx) DIE("Failed to open device");

        ibv_query_port(ctx, IB_PORT, &port_attr);
        ibv_query_gid(ctx, IB_PORT, GID_IDX, &gid);
        lid = port_attr.lid;

        pd = ibv_alloc_pd(ctx);
        cq = ibv_create_cq(ctx, 10, NULL, NULL, 0);
        
        // Register Memory
        mr = ibv_reg_mr(pd, buf, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr) DIE("MR Reg failed");

        // Create QP
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.qp_type = IBV_QPT_RC; // Reliable Connection
        qp_init_attr.cap.max_send_wr = 10;
        qp_init_attr.cap.max_recv_wr = 10;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        
        qp = ibv_create_qp(pd, &qp_init_attr);
        if (!qp) DIE("QP Create failed");
        
        change_qp_state_init();
    }

    void change_qp_state_init() {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_INIT;
        attr.port_num = IB_PORT;
        attr.pkey_index = 0;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            DIE("Failed to modify QP to INIT");
    }

    void change_qp_state_rtr() {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024; // Safe default for RoCE
        attr.dest_qp_num = remote_qpn;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = remote_gid;
        attr.ah_attr.grh.sgid_index = GID_IDX;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = IB_PORT;

        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | 
                                     IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            DIE("Failed to modify QP to RTR");
    }

    void change_qp_state_rts() {
        struct ibv_qp_attr attr = {};
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | 
                                     IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            DIE("Failed to modify QP to RTS");
    }
};

// --- MAIN BENCHMARK LOGIC ---

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./bench_net <server|client> [server_ip]" << std::endl;
        return 1;
    }
    bool is_server = (std::string(argv[1]) == "server");
    std::string ip = is_server ? "0.0.0.0" : argv[2];

    size_t BUFFER_SIZE = 100 * 1024 * 1024; // 100 MB
    int ITERATIONS = 10;

    // Alloc Host Memory (Pinned for best RoCE performance)
    void *h_buffer;
    CUDA_CHECK(cudaMallocHost(&h_buffer, BUFFER_SIZE));
    memset(h_buffer, 1, BUFFER_SIZE);

    // Alloc GPU Memory
    void *d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, BUFFER_SIZE));

    // 1. Establish TCP Connection (Control Plane)
    int tcp_sock = is_server ? setup_tcp_server() : setup_tcp_client(ip.c_str());
    std::cout << "[Info] TCP Connection Established." << std::endl;

    // 2. Setup RDMA
    RDMAContext rdma;
    rdma.init(h_buffer, BUFFER_SIZE);

    // Exchange QP Info
    ExchangeData local_ex = { (uint32_t)rdma.lid, rdma.qp->qp_num, rdma.gid };
    ExchangeData remote_ex;
    
    tcp_send_data(tcp_sock, &local_ex, sizeof(ExchangeData));
    tcp_recv_data(tcp_sock, &remote_ex, sizeof(ExchangeData));
    
    rdma.remote_lid = remote_ex.lid;
    rdma.remote_qpn = remote_ex.qp_num;
    rdma.remote_gid = remote_ex.gid;

    rdma.change_qp_state_rtr();
    rdma.change_qp_state_rts();
    std::cout << "[Info] RDMA Queue Pairs Ready." << std::endl;

    // ==========================================
    // BENCHMARK 1: TCP + GPU Load
    // ==========================================
    std::cout << "\n--- Starting TCP Benchmark ---" << std::endl;
    // Barrier
    char sync = 'x';
    tcp_send_data(tcp_sock, &sync, 1); tcp_recv_data(tcp_sock, &sync, 1);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; i++) {
        if (!is_server) {
            // Client sends
            tcp_send_data(tcp_sock, h_buffer, BUFFER_SIZE);
            // Simulate waiting for ack (simple app-level sync)
            tcp_recv_data(tcp_sock, &sync, 1); 
        } else {
            // Server receives
            tcp_recv_data(tcp_sock, h_buffer, BUFFER_SIZE);
            // Upload to GPU
            CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, BUFFER_SIZE, cudaMemcpyHostToDevice));
            // Send Ack
            tcp_send_data(tcp_sock, &sync, 1);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double tcp_time = std::chrono::duration<double>(end - start).count();
    
    if (is_server) {
        double bw = (double)BUFFER_SIZE * ITERATIONS / tcp_time / 1e9; // GB/s
        std::cout << "TCP + GPU Load Time: " << tcp_time << " s" << std::endl;
        std::cout << "Throughput: " << bw << " GB/s" << std::endl;
    }

    // ==========================================
    // BENCHMARK 2: Soft-RoCE + GPU Load
    // ==========================================
    std::cout << "\n--- Starting Soft-RoCE Benchmark ---" << std::endl;
    // Barrier
    tcp_send_data(tcp_sock, &sync, 1); tcp_recv_data(tcp_sock, &sync, 1);

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; i++) {
        if (!is_server) {
            // Client: Post Send
            struct ibv_sge sge;
            sge.addr = (uint64_t)h_buffer;
            sge.length = BUFFER_SIZE;
            sge.lkey = rdma.mr->lkey;

            struct ibv_send_wr wr, *bad_wr;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = i;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_SEND;
            wr.send_flags = IBV_SEND_SIGNALED;

            if (ibv_post_send(rdma.qp, &wr, &bad_wr)) DIE("Post Send failed");

            // Wait for completion
            struct ibv_wc wc;
            while(ibv_poll_cq(rdma.cq, 1, &wc) == 0);
            if (wc.status != IBV_WC_SUCCESS) DIE("Send Completion Error");

            // Wait for Server Ack via TCP (simplifies sync)
            tcp_recv_data(tcp_sock, &sync, 1);

        } else {
            // Server: Post Recv
            struct ibv_sge sge;
            sge.addr = (uint64_t)h_buffer;
            sge.length = BUFFER_SIZE;
            sge.lkey = rdma.mr->lkey;

            struct ibv_send_wr recv_wr; // unused for recv, but needed conceptually
            struct ibv_recv_wr wr, *bad_wr;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id = i;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            
            if (ibv_post_recv(rdma.qp, &wr, &bad_wr)) DIE("Post Recv failed");

            // Wait for data arrival
            struct ibv_wc wc;
            while(ibv_poll_cq(rdma.cq, 1, &wc) == 0);
            if (wc.status != IBV_WC_SUCCESS) DIE("Recv Completion Error");

            // Upload to GPU
            CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, BUFFER_SIZE, cudaMemcpyHostToDevice));

            // Send Ack via TCP
            tcp_send_data(tcp_sock, &sync, 1);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double roce_time = std::chrono::duration<double>(end - start).count();

    if (is_server) {
        double bw = (double)BUFFER_SIZE * ITERATIONS / roce_time / 1e9; // GB/s
        std::cout << "RoCE + GPU Load Time: " << roce_time << " s" << std::endl;
        std::cout << "Throughput: " << bw << " GB/s" << std::endl;
        
        std::cout << "\n-----------------------------------" << std::endl;
        std::cout << "Results Summary:" << std::endl;
        std::cout << "TCP  Speed: " << (double)BUFFER_SIZE * ITERATIONS / tcp_time / 1e9 << " GB/s" << std::endl;
        std::cout << "RoCE Speed: " << (double)BUFFER_SIZE * ITERATIONS / roce_time / 1e9 << " GB/s" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    // Cleanup
    cudaFreeHost(h_buffer);
    cudaFree(d_buffer);
    close(tcp_sock);
    return 0;
}