import SwiftUI
import AppKit

@main
struct TraceTracker: App {
    @StateObject private var server = ServerManager()

    var body: some Scene {
        MenuBarExtra {
            Button(server.isRunning ? "Open in Browser" : "Start & Open") {
                if server.isRunning {
                    server.openBrowser()
                } else {
                    server.start()
                }
            }
            .disabled(server.isStarting)

            Divider()

            Text(server.statusText).font(.caption)

            if server.isRunning {
                Divider()
                Button("Stop Server") { server.stop() }
            }

            Divider()

            Button("Preferences...") { server.showPreferences() }
            Button("Quit") {
                server.stop()
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
        } label: {
            Image(systemName: server.isRunning ? "point.3.filled.connected.trianglepath.dotted" : "point.3.connected.trianglepath.dotted")
        }
    }
}

class ServerManager: ObservableObject {
    @Published var isRunning = false
    @Published var isStarting = false
    @Published var statusText = "Ready"

    private var process: Process?

    private var projectPath: String {
        get {
            if let p = UserDefaults.standard.string(forKey: "projectPath"), !p.isEmpty {
                return p
            }
            return defaultPath
        }
        set { UserDefaults.standard.set(newValue, forKey: "projectPath") }
    }

    private var defaultPath: String {
        let bundlePath = Bundle.main.bundleURL.path
        let parent = URL(fileURLWithPath: bundlePath).deletingLastPathComponent().path
        let sibling = parent + "/viewer"
        if FileManager.default.fileExists(atPath: sibling) {
            return sibling
        }
        return NSHomeDirectory() + "/Desktop/local-reasoner/trace-trek/viewer"
    }

    func start() {
        guard !isStarting else { return }

        let path = projectPath
        guard FileManager.default.fileExists(atPath: path) else {
            statusText = "Project path not found — set in Preferences"
            return
        }

        isStarting = true
        statusText = "Starting..."

        let proc = Process()
        proc.currentDirectoryURL = URL(fileURLWithPath: path)
        proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
        proc.arguments = ["-l", "-c", "exec npm run dev"]

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty { return }
            if let text = String(data: data, encoding: .utf8) {
                if text.contains("Local:") || text.contains("localhost") {
                    DispatchQueue.main.async {
                        guard let self = self, self.isStarting else { return }
                        self.isRunning = true
                        self.isStarting = false
                        self.statusText = "Running"
                        self.openBrowser()
                    }
                }
            }
        }

        self.process = proc

        DispatchQueue.global().async { [weak self] in
            do {
                try proc.run()
                proc.waitUntilExit()
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    if self.isRunning {
                        self.statusText = "Server stopped"
                        self.isRunning = false
                    } else if self.isStarting {
                        self.statusText = "Failed to start — check terminal for errors"
                        self.isStarting = false
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self?.statusText = "Error: \(error.localizedDescription)"
                    self?.isStarting = false
                }
            }
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        isRunning = false
        isStarting = false
        statusText = "Stopped"
    }

    func openBrowser() {
        guard let url = URL(string: "http://localhost:5173") else { return }
        NSWorkspace.shared.open(url)
    }

    func showPreferences() {
        let alert = NSAlert()
        alert.messageText = "Trace Anatomy Project Path"
        alert.informativeText = "Path to the viewer folder:"

        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 420, height: 22))
        field.stringValue = projectPath
        alert.accessoryView = field
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")

        if let window = NSApp.windows.first {
            alert.beginSheetModal(for: window) { [weak self] resp in
                if resp == .alertFirstButtonReturn {
                    self?.projectPath = field.stringValue
                    self?.statusText = "Path updated"
                }
            }
        } else {
            let resp = alert.runModal()
            if resp == .alertFirstButtonReturn {
                projectPath = field.stringValue
                statusText = "Path updated"
            }
        }
    }
}
