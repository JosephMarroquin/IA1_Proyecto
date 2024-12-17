let intentsData = null;
let model = null;
let words = [];
let labels = [];

// Función para cargar los datos y el modelo
async function loadData() {
    const response = await fetch('intents.json');
    intentsData = await response.json();

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

    const training = [];
    const output = [];

    const outEmpty = Array(labels.length).fill(0);

    docsX.forEach((doc, index) => {
        const bag = words.map(word => (doc.includes(word) ? 1 : 0));
        const outputRow = outEmpty.slice();
        outputRow[labels.indexOf(docsY[index])] = 1;
        training.push(bag);
        output.push(outputRow);
    });

    model = await trainModel(training, output);

    // Guardar el modelo en formato JSON
    await saveModel();

    alert('Modelo entrenado y guardado correctamente en un archivo JSON.');
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

// Entrenamiento del modelo
async function trainModel(trainingData, outputData) {
    const inputShape = trainingData[0].length;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, inputShape: [inputShape], activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));  // Dropout para evitar overfitting
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));

    model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

    const xs = tf.tensor(trainingData);
    const ys = tf.tensor(outputData);

    await model.fit(xs, ys, {
        epochs: 1000,
        batchSize: 8,
        verbose: 1
    });

    return model;
}

// Guardar el modelo en formato JSON
async function saveModel() {
    const saveResult = await model.save('downloads://chatbot_model');
    console.log('Modelo guardado:', saveResult);
}

// Cargar el modelo al iniciar la página
window.onload = loadData;
