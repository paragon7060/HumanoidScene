// Standalone host for the NVIDIA CloudXR 6.x Runtime Management API.
// The NVIDIA SDK is supplied separately; no SDK binaries are redistributed.
#include <arpa/inet.h>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include "cxrServiceAPI.h"

static volatile std::sig_atomic_t interrupted = 0;
static void stop_signal(int) { interrupted = 1; }

static void check(nv_cxr_result_t result, const char* operation) {
    if (result != NV_CXR_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
}

static std::string read_pem(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read PEM file: " + path);
    std::ostringstream contents;
    contents << file.rdbuf();
    const auto value = contents.str();
    if (value.empty() || value.size() > 65536 || value.find("-----BEGIN ") == std::string::npos)
        throw std::runtime_error("Invalid PEM file: " + path);
    return value;
}

int main(int argc, char** argv) {
    nv_cxr_service* service = nullptr;
    bool started = false;
    int exit_code = 0;
    try {
        std::string host = "127.0.0.1", certificate, key;
        int port = 49100, media_port = 47998, seconds = 0;
        bool check_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--check") { check_only = true; continue; }
            if (arg == "--help") {
                std::cout << "Usage: run_cloudxr_runtime.sh [--check] [--host IPv4] "
                          << "[--port 49100] [--media-port 47998] [--seconds N] "
                          << "[--certificate PEM --key PEM]\n"
                          << "--host sets the advertised endpoint. SDK signaling may bind all interfaces.\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--host") host = value;
            else if (arg == "--certificate") certificate = value;
            else if (arg == "--key") key = value;
            else if (arg == "--port" || arg == "--media-port" || arg == "--seconds") {
                size_t used = 0;
                int number = std::stoi(value, &used);
                if (used != value.size()) throw std::runtime_error("Invalid number: " + value);
                if (arg == "--port") port = number;
                else if (arg == "--media-port") media_port = number;
                else seconds = number;
            } else throw std::runtime_error("Unknown option: " + arg);
        }
        in_addr address{};
        if (inet_pton(AF_INET, host.c_str(), &address) != 1)
            throw std::runtime_error("--host must be an IPv4 address");
        if (port < 1024 || port > 65535 || media_port < 1024 || media_port > 65535 || seconds < 0)
            throw std::runtime_error("Ports must be 1024..65535 and seconds nonnegative");
        if (certificate.empty() != key.empty())
            throw std::runtime_error("--certificate and --key must be supplied together");

        uint32_t major, minor, patch;
        nv_cxr_get_runtime_version(&major, &minor, &patch);
        std::cout << "CloudXR Runtime " << major << '.' << minor << '.' << patch << std::endl;
        nv_cxr_get_library_api_version(&major, &minor, &patch);
        std::cout << "Management API " << major << '.' << minor << '.' << patch << std::endl;
        if (major != NV_CXR_API_VERSION_MAJOR)
            throw std::runtime_error("SDK header/library API major version mismatch");
        if (check_only) return 0;

        std::signal(SIGINT, stop_signal);
        std::signal(SIGTERM, stop_signal);
        check(nv_cxr_service_create(&service), "create");
        auto set_string = [&](const char* name, const std::string& value) {
            check(nv_cxr_service_set_string_property(service, name, std::strlen(name),
                                                     value.data(), value.size()), name);
        };
        auto set_int = [&](const char* name, int value) {
            check(nv_cxr_service_set_int64_property(service, name, std::strlen(name), value), name);
        };
        set_string("device-profile", "auto-webrtc");
        set_string("endpoint-ip", host);
        set_int("server-port", port);
        set_int("media-port", media_port);
        if (!certificate.empty()) {
            // Runtime 6.2.1 expects PEM contents, despite the online reference
            // describing these properties as paths. Never log the private key.
            set_string("certificate-pem", read_pem(certificate));
            set_string("key-pem", read_pem(key));
        }
        check(nv_cxr_service_set_boolean_property(service, "audio-streaming", 15, false), "audio-streaming");
        check(nv_cxr_service_start(service), "start");
        started = true;
        std::cout << "[READY] CloudXR " << (certificate.empty() ? "ws://" : "wss://")
                  << host << ':' << port << "; media UDP " << media_port << std::endl;
        std::cout << "Connect the Quest client before starting the OpenXR collector.\n"
                  << "[SECURITY] The SDK may listen on all interfaces; use only on a trusted LAN.\n"
                  << "Service readiness does not verify encoding or headset tracking." << std::endl;
        const auto start = std::chrono::steady_clock::now();
        while (!interrupted && (seconds == 0 || std::chrono::steady_clock::now() - start < std::chrono::seconds(seconds))) {
            nv_cxr_event_t event{};
            check(nv_cxr_service_poll_event(service, &event), "poll_event");
            switch (event.type) {
                case NV_CXR_EVENT_CLOUDXR_CLIENT_CONNECTED: std::cout << "[CLIENT] Connected" << std::endl; break;
                case NV_CXR_EVENT_CLOUDXR_CLIENT_DISCONNECTED: std::cout << "[CLIENT] Disconnected" << std::endl; break;
                case NV_CXR_EVENT_OPENXR_APP_CONNECTED: std::cout << "[OPENXR] App connected" << std::endl; break;
                case NV_CXR_EVENT_OPENXR_APP_DISCONNECTED: std::cout << "[OPENXR] App disconnected" << std::endl; break;
                default: break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << std::endl;
        exit_code = 1;
    }
    if (started) {
        const auto result = nv_cxr_service_stop(service);
        if (result != NV_CXR_SUCCESS && result != NV_CXR_SERVICE_NOT_STARTED) exit_code = 1;
        const auto joined = nv_cxr_service_join(service);
        if (joined != NV_CXR_SUCCESS && joined != NV_CXR_SERVICE_NOT_STARTED) exit_code = 1;
    }
    if (service) nv_cxr_service_destroy(service);
    return exit_code;
}
