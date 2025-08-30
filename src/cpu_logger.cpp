// cpu_logger.cpp (container-aware, 10ms sampling default)
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>     // sysconf
#include <algorithm>
#include <cstdint>
#include <cmath>        // std::floor
#include <ctime>        // gmtime_r

struct CpuTimes {
    unsigned long long user=0, nice_=0, system=0, idle=0, iowait=0,
                       irq=0, softirq=0, steal=0, guest=0, guest_nice=0;
};

static bool read_text_file(const char* path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Parse "0-3,9,12-13" -> đếm số CPU
static int count_cpuset_list(const std::string& s) {
    int count = 0;
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]==',')) ++i;
        if (i >= n) break;
        char* endp = nullptr;
        long a = strtol(&s[i], &endp, 10);
        if (endp == &s[i]) { ++i; continue; }
        i = size_t(endp - s.c_str());
        long b = -1;
        if (i < n && s[i] == '-') {
            ++i;
            b = strtol(&s[i], &endp, 10);
            if (endp != &s[i]) i = size_t(endp - s.c_str());
        }
        if (b >= a && b != -1) count += int(b - a + 1);
        else count += 1;
    }
    return count;
}

// Lấy số CPU effective (cpuset/quota), fallback sysconf/hardware_concurrency
static int get_nproc_visible() {
    std::string s;
    // cgroup v2
    if (read_text_file("/sys/fs/cgroup/cpuset.cpus.effective", s) && !s.empty()) {
        int m = count_cpuset_list(s);
        if (m > 0) return m;
    }
    // cgroup v1
    if (read_text_file("/sys/fs/cgroup/cpuset/cpuset.cpus", s) && !s.empty()) {
        int m = count_cpuset_list(s);
        if (m > 0) return m;
    }
    // quota cgroup v2: cpu.max => "<quota> <period>" hoặc "max <period>"
    std::string cpu_max;
    if (read_text_file("/sys/fs/cgroup/cpu.max", cpu_max) && !cpu_max.empty()) {
        long long quota=0, period=100000; // usec
        if (cpu_max.rfind("max", 0) != 0) {
            if (sscanf(cpu_max.c_str(), "%lld %lld", &quota, &period) == 2 && quota > 0 && period > 0) {
                double cpus = double(quota) / double(period);
                int m = (int)std::max(1.0, std::floor(cpus + 1e-9));
                if (m > 0) return m;
            }
        }
    }
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return (int)n;
#endif
    unsigned hc = std::thread::hardware_concurrency();
    return hc ? (int)hc : 1;
}

// ==== cgroup-based container CPU usage ====

// cgroup v2: /sys/fs/cgroup/cpu.stat -> "usage_usec <num>"
static bool read_cgroup_v2_usage_usec(uint64_t& usec_out) {
    std::string s;
    if (!read_text_file("/sys/fs/cgroup/cpu.stat", s)) return false;
    std::istringstream iss(s);
    std::string key; uint64_t val=0;
    while (iss >> key >> val) {
        if (key == "usage_usec") { usec_out = val; return true; }
    }
    return false;
}

// cgroup v1: cpuacct.usage (ns)
static bool read_cgroup_v1_usage_ns(uint64_t& ns_out) {
    const char* paths[] = {
        "/sys/fs/cgroup/cpuacct/cpuacct.usage",
        "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage",
        "/sys/fs/cgroup/cpuacct.usage",
    };
    for (const char* p : paths) {
        std::ifstream f(p);
        if (f) {
            uint64_t v=0; f >> v;
            if (f.good()) { ns_out = v; return true; }
        }
    }
    return false;
}

// Fallback: host-wide từ /proc/stat
static bool read_cpu_aggregate(CpuTimes& t) {
    std::ifstream f("/proc/stat");
    if (!f) return false;
    std::string cpu;
    f >> cpu;
    if (cpu != "cpu") return false;
    f >> t.user >> t.nice_ >> t.system >> t.idle >> t.iowait
      >> t.irq >> t.softirq >> t.steal >> t.guest >> t.guest_nice;
    return true;
}

static double cpu_percent_host(double interval_sec) {
    CpuTimes a,b;
    if (!read_cpu_aggregate(a)) return 0.0;
    std::this_thread::sleep_for(std::chrono::milliseconds((int)(interval_sec*1000)));
    if (!read_cpu_aggregate(b)) return 0.0;

    auto idleA = a.idle + a.iowait;
    auto idleB = b.idle + b.iowait;
    auto nonIdleA = a.user + a.nice_ + a.system + a.irq + a.softirq + a.steal;
    auto nonIdleB = b.user + b.nice_ + b.system + b.irq + b.softirq + b.steal;
    auto totalA = idleA + nonIdleA;
    auto totalB = idleB + nonIdleB;

    double totald = double(totalB - totalA);
    double idled  = double(idleB - idleA);
    if (totald <= 0) return 0.0;
    return 100.0 * (totald - idled) / totald;
}

// %CPU container (0..100), ưu tiên cgroup; fallback host
static double cpu_percent_container(double interval_sec, int ncpus_eff, std::string* method = nullptr) {
    if (ncpus_eff <= 0) ncpus_eff = 1;

    // cgroup v2
    uint64_t u0=0, u1=0;
    if (read_cgroup_v2_usage_usec(u0)) {
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(interval_sec*1000)));
        if (read_cgroup_v2_usage_usec(u1) && u1 >= u0) {
            double du_sec = double(u1 - u0) / 1e6; // usec -> sec
            double pct = 100.0 * du_sec / (interval_sec * ncpus_eff);
            if (method) *method = "cgroupv2";
            if (pct < 0) pct = 0; if (pct > 100) pct = 100;
            return pct;
        }
    }

    // cgroup v1
    uint64_t n0=0, n1=0;
    if (read_cgroup_v1_usage_ns(n0)) {
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(interval_sec*1000)));
        if (read_cgroup_v1_usage_ns(n1) && n1 >= n0) {
            double du_sec = double(n1 - n0) / 1e9; // ns -> sec
            double pct = 100.0 * du_sec / (interval_sec * ncpus_eff);
            if (method) *method = "cgroupv1";
            if (pct < 0) pct = 0; if (pct > 100) pct = 100;
            return pct;
        }
    }

    // Fallback: host-wide
    if (method) *method = "procstat";
    return cpu_percent_host(interval_sec);
}

static std::string now_iso_utc() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t tt = system_clock::to_time_t(now);
    char buf[32];
    std::tm tm{};
    gmtime_r(&tt, &tm);
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

int main(int argc, char** argv) {
    double interval = 0.01;            // === mặc định 10ms ===
    double duration = 300.0;            // tổng thời gian (s)
    std::string outfile = "./logs/cpu.csv";
    std::string tag = "container";

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--interval") && i+1 < argc) interval = atof(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i+1 < argc) duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--outfile")  && i+1 < argc) outfile  = argv[++i];
        else if (!strcmp(argv[i], "--tag")      && i+1 < argc) tag      = argv[++i];
    }

    std::ofstream out(outfile);
    if (!out) {
        std::cerr << "Cannot open outfile: " << outfile << "\n";
        return 1;
    }

    out << "timestamp_iso,t_sec,tag,cpu_percent,n_cpus_visible\n";
    out.flush();

    const int n_eff = get_nproc_visible();
    const auto start = std::chrono::steady_clock::now();

    // 10ms sampling → I/O nhiều; cân nhắc để ổn định
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        const double t = std::chrono::duration<double>(now - start).count();
        if (t > duration) break;

        std::string method;
        double cpu = cpu_percent_container(interval, n_eff, &method);
        out << now_iso_utc() << ',' << t << ',' << tag << ',' << cpu << ',' << n_eff << "\n";
        out.flush();
    }
    return 0;
}
