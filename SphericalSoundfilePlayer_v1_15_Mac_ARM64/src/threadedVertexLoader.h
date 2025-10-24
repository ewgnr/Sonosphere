#pragma once

#include "ofMain.h"
#include "sfPlayer.h"

struct VertexLoadTask {
    int vertexIndex;
    std::string filePath;
};

class ThreadedVertexLoader : public ofThread {
public:
    void start(const std::vector<VertexLoadTask>& newTasks) {
        std::lock_guard<std::mutex> lock(mutex);

        if (isThreadRunning()) {
            stopThread();
            waitForThread();
        }

        tasks = newTasks;
        results.clear();
        loading = true;

        startThread();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex);
        if (isThreadRunning()) {
            stopThread();
            waitForThread();
            loading = false;
        }
    }

    void threadedFunction() override {
        std::vector<std::pair<int, std::shared_ptr<sfPlayer>>> tempResults;

        for (const auto& task : tasks) {
            auto player = std::make_shared<sfPlayer>();

            if (player->load(ofToDataPath(task.filePath, true))) {
                tempResults.emplace_back(task.vertexIndex, player);
            }
            else {
                tempResults.emplace_back(task.vertexIndex, nullptr);
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            results = std::move(tempResults);
            loading = false;
        }
    }

    std::vector<std::pair<int, std::shared_ptr<sfPlayer>>> getResults() {
        std::lock_guard<std::mutex> lock(mutex);
        return results;
    }

    bool isLoading() const {
        return loading.load();
    }

private:
    std::vector<VertexLoadTask> tasks;
    std::vector<std::pair<int, std::shared_ptr<sfPlayer>>> results;
    mutable std::mutex mutex;
    std::atomic<bool> loading = false;
};
