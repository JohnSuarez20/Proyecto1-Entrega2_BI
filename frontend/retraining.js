// retraining.js
function retrainModel() {
    const fileInput = document.getElementById('retrain-file');
    if (fileInput.files.length === 0) {
        alert('Por favor, selecciona un archivo CSV.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('http://127.0.0.1:8000/reentrenamiento/', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // Asumiendo que el backend retorna las métricas
        document.getElementById('exactitud').innerText = data.exactitud;
        document.getElementById('precision').innerText = data.precision;
        document.getElementById('recall').innerText = data.recall;
        document.getElementById('f1_score').innerText = data.f1_score;
    })
    .catch(error => {
        console.error('Error en el reentrenamiento:', error);
        alert('Ocurrió un error durante el reentrenamiento.');
    });
}
