const fs = require("fs");

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

const wasm = fs.readFileSync("target/wasm32-unknown-unknown/release/write_frames.wasm");
(async () => {
    instance = (await WebAssembly.instantiate(wasm, importObject)).instance;
    instance.exports.main();
})();
