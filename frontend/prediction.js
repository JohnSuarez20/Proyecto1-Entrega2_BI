async function sendManualPrediction() {
    const count = document.getElementById('opinion-count').value;
    let opinions = [];
    for (let i = 1; i <= count; i++) {
        const opinion = document.getElementById(`opinion${i}`).value;
        opinions.push(opinion);
    }

    const response = await fetch('http://localhost:8000/prediccion/', {
        method: 'POST',
        body: JSON.stringify({ Textos_espanol: opinions }),
        headers: {
            'Content-Type': 'application/json',
        }
    });

    if (response.ok) {
        const result = await response.json();
        displayPredictionResult(result.resultado);
    } else {
        alert('Error al realizar la predicción');
        console.error("Error response:", await response.json());
    }
}


// Función para mostrar los resultados de la predicción manual
function displayPredictionResult(result) {
    const tableBody = document.querySelector('#manual-prediction-result tbody');
    tableBody.innerHTML = '';  // Limpia la tabla

    result.forEach((item, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${item.Textos_espanol}</td>
            <td>${item.prediccion}</td>
            <td>${item.probabilidad.toFixed(2)}</td>
        `;
        tableBody.appendChild(row);
    });
}

// Función para enviar un archivo para la predicción masiva
async function sendPredictionFile() {
    const fileInput = document.getElementById('pred-file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('http://localhost:8000/prediccion/', {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        const result = await response.json();
        displayPredictionResult(result.resultado);
    } else {
        alert('Error al realizar la predicción con archivo');
    }
}

async function predictManual() {
    const texto = document.getElementById("manual-text").value;  // Obtener el texto de un input
    const response = await fetch("http://localhost:8000/prediccion/manual/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ texto: texto })  // Enviar el texto como un objeto JSON
    });

    if (response.ok) {
        const data = await response.json();
        document.getElementById("predictions").innerHTML = JSON.stringify(data);
    } else {
        console.error("Error en la predicción:", response.statusText);
    }
}


