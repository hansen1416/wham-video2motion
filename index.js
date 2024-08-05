const fs = require('fs');
const path = require('path');

function read_data (filename) {
    // Read from ArrayBuffer
    const buffer = fs.readFileSync(filename);
    const data = new Float32Array(buffer);
    console.log(data); // Output: Float32Array(3) [1.1, 2.2, 3.3]

    console.log(data.length)
    return data
}

const pose = path.join('results', 'madfit1', 'pose.bin');
const pose_world = path.join('results', 'madfit1', 'pose_world.bin');
const trans = path.join('results', 'madfit1', 'trans.bin');
const trans_world = path.join('results', 'madfit1', 'trans_world.bin');


read_data(pose);
read_data(pose_world);
read_data(trans)
read_data(trans_world)

