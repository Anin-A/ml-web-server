const predictClassification = require('../services/inferenceService');
const storeData = require('../services/storeData');  // Impor fungsi storeData
const crypto = require('crypto');

async function postPredictHandler(request, h) {
    const { image } = request.payload;
    const { model } = request.server.app;

    const { confidenceScore, label, explanation, suggestion } = await predictClassification(model, image);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();

    const data = {
        "id": id,
        "result": label,
        "explanation": explanation,
        "suggestion": suggestion,
        "confidenceScore": confidenceScore,
        "createdAt": createdAt
    };

    if (label === 'Cancer') {
        data.explanation = explanation;
        data.confidenceScore = confidenceScore;
    }

    // Simpan data ke Firestore
    await storeData(id, JSON.parse(JSON.stringify(data)))

    const response = h.response({
        status: 'success',
        message: 'Model is predicted successfully',
        data
    });
    response.code(201);
    return response;
}

module.exports = postPredictHandler;
