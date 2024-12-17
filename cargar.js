let model = null;
let words = [];
let labels = [];
let intentsData = null;
let context = {}; // Contexto de la conversación

// Cargar los datos de los intents (este archivo no cambia)
async function loadData() {
    const response = await fetch('intents.json');
    intentsData = await response.json();

    // Se deben cargar los datos para las predicciones (palabras y etiquetas)
    const docsX = [];
    const docsY = [];
    intentsData.intents.forEach(intent => {
        intent.patterns.forEach(pattern => {
            const wordsInPattern = tokenizeAndStem(pattern);
            words.push(...wordsInPattern);
            docsX.push(wordsInPattern);
            docsY.push(intent.tag);
        });

        if (!labels.includes(intent.tag)) {
            labels.push(intent.tag);
        }
    });

    words = [...new Set(words)].sort();
    labels = labels.sort();
}

// Tokenización y lematización
function tokenizeAndStem(sentence) {
    const wordsInSentence = sentence
        .toLowerCase()
        .replace(/[^\w\s]/g, '')  // Elimina caracteres no alfanuméricos
        .split(/\s+/);  // Separa por espacios

    // Lematización y filtrado de stop words
    const stopWords = ['de', 'la', 'el', 'y', 'a', 'en', 'es', 'que', 'por', 'con'];
    return wordsInSentence.map(word => stem(word))
                          .filter(word => !stopWords.includes(word));
}

// Stemming de las palabras
function stem(word) {
    if (word.endsWith('es')) {
        return word.slice(0, -2);
    } else if (word.endsWith('ed')) {
        return word.slice(0, -2);
    } else if (word.endsWith('ing')) {
        return word.slice(0, -3);
    } else {
        return word;
    }
}

// Cargar el modelo previamente guardado
async function loadModel() {
    // Cargar el modelo de los archivos JSON y BIN que están en el directorio principal
    model = await tf.loadLayersModel('chatbot_model.json');  // Modelo JSON
    console.log('Modelo cargado exitosamente');
}

// Predicción de la respuesta
async function predictResponse(input) {
    const bag = words.map(word => (input.includes(word) ? 1 : 0));
    const prediction = await model.predict(tf.tensor([bag]));
    const predictedIndex = prediction.argMax(1).dataSync()[0];
    const predictedTag = labels[predictedIndex];

    const intent = intentsData.intents.find(intent => intent.tag === predictedTag);
    return getResponse(intent.responses);
}

// Obtener respuesta aleatoria
function getResponse(responses) {
    return responses[Math.floor(Math.random() * responses.length)];
}

// Enviar mensaje (función para enviar mensaje del usuario y mostrar la respuesta)
async function sendMessage() {
    const userInput = document.getElementById('user_input').value;
    if (userInput.toLowerCase() === 'quit') {
        return;
    }

    // Mostrar mensaje del usuario
    document.getElementById('chat').innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;
    document.getElementById('user_input').value = '';

    // Obtener respuesta del bot
    const response = await predictResponse(userInput);

    // Mostrar respuesta del bot
    document.getElementById('chat').innerHTML += `<div><strong>Bot:</strong> ${response}</div>`;
    document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
}

// Inicializar el chat
window.onload = async () => {
    // Cargar datos y modelo
    await loadData();
    await loadModel();  // Cargar el modelo previamente entrenado

    // Activar interfaz de chat
    document.getElementById('loading').style.display = 'none';  // Ocultar mensaje de carga
    document.getElementById('user_input').disabled = false;  // Habilitar el campo de entrada
    document.querySelector('button').disabled = false;  // Habilitar el botón de envío
};
