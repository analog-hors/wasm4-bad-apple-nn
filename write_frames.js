const fs = require("fs");
const child_process = require("child_process");

function buildModule() {
    child_process.execFileSync(
        "cargo",
        [
            "build",
            "-p", "write-frames",
            "--release",
            "--config", "write_frames_config.toml",
        ],
        { stdio: "inherit" },
    );
    const buffer = fs.readFileSync("target/wasm32-unknown-unknown/release/write_frames.wasm");
    return new WebAssembly.Module(buffer);
}

function runModule(module) {
    let instance;
    const getByteSlice = (ptr, size) => {
        const mem = instance.exports.memory.buffer;
        return new Uint8Array(mem.slice(ptr, ptr + size));
    };
    const importObject = {
        env: {
            time: () => performance.now(),
            print: (messagePtr, messageSize) => {
                const bytes = getByteSlice(messagePtr, messageSize);
                const message = Buffer.from(bytes, "utf-8").toString();
                console.log(message);
            },
            write_file: (pathPtr, pathSize, bufferPtr, bufferSize) => {
                const path = getByteSlice(pathPtr, pathSize);
                const buffer = getByteSlice(bufferPtr, bufferSize);
                fs.writeFileSync(path, buffer);
            },
        }
    };

    instance = new WebAssembly.Instance(module, importObject);
    instance.exports.main();
}

runModule(buildModule());
