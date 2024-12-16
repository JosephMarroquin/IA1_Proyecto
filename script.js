let intentsData = null;
let model = null;
let words = [];
let labels = [];

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
    document.getElementById('loading').style.display = 'none';  
    document.getElementById('user_input').disabled = false;  
    document.querySelector('button').disabled = false;  
}

function tokenizeAndStem(sentence) {
    const wordsInSentence = sentence
        .toLowerCase()
        .replace(/[^\w\s]/g, '')  
        .split(/\s+/);  

    return wordsInSentence.map(word => stem(word)).filter(word => word !== "?");
}

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

async function trainModel(trainingData, outputData) {
    const inputShape = trainingData[0].length;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 8, inputShape: [inputShape], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
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

async function predictResponse(input) {
    const bag = words.map(word => (input.includes(word) ? 1 : 0));
    const prediction = await model.predict(tf.tensor([bag]));
    const predictedIndex = prediction.argMax(1).dataSync()[0];
    const predictedTag = labels[predictedIndex];

    const intent = intentsData.intents.find(intent => intent.tag === predictedTag);
    return intent.responses[Math.floor(Math.random() * intent.responses.length)];
}

async function sendMessage() {
    const userInput = document.getElementById('user_input').value;
    if (userInput.toLowerCase() === 'quit') {
        return;
    }

    const response = await predictResponse(userInput);
    document.getElementById('chat').innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;
    document.getElementById('chat').innerHTML += `<div><strong>Bot:</strong> ${response}</div>`;
}

window.onload = loadData;