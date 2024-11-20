const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
    try {
        // Konversi data gambar (image) menjadi tensor
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        // Dapatkan prediksi dari model
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        // Dapatkan hasil prediksi dengan nilai tertinggi
        const classResult = tf.argMax(prediction, 1).dataSync()[0];

        // Berikan penjelasan dan saran berdasarkan hasil prediksi
        let explanation, suggestion, label;
        if (confidenceScore === 100) {
            suggestion = "Segera periksa ke dokter!";
            label = "Cancer"
        } else {
            suggestion = "Penyakit kanker tidak terdeteksi.";
            label = "Non-cancer"
        }

        return {suggestion, label };
    } catch (error) {
        // Tangani kesalahan input
        throw new InputError(`Terjadi kesalahan input: ${error.message}`);
    }
}

module.exports = predictClassification;
