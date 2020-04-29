const pokemonClasses = [
  'Normal',
  'Fighting',
  'Flying',
  'Poison',
  'Ground',
  'Rock',
  'Bug',
  'Ghost',
  'Fire',
  'Water',
  'Grass',
  'Electric',
  'Psychic',
  'Ice',
  'Dragon',
  'Fairy'
];

var generator, classifier;

const start = async function(a, b) {
  generator = await tf.loadLayersModel('public/model/generator/model.json');
  classifier = await tf.loadLayersModel('public/model/classifier/model.json');
}

start();

$(document).ready(function() { 
  $('#status').html('Generator is ready.');
})

var pokemonNumber = 30;

for (let i = 0; i < pokemonNumber; i++) {
  $('#generatedImage').append(`<canvas id="${i}" class="image" data-toggle="tooltip"></canvas>`)
}

const generatePokemon = async function() {
  for (let i = 0; i < pokemonNumber; i++) {
    let noise = tf.randomNormal([1, 100]);
    let generatedImage = generator.predict(noise)

    let imageData = tf.reshape(generatedImage, [28, 28, 3]);
    imageData = imageData.mul(tf.scalar(0.5));
    imageData = imageData.add(tf.scalar(0.5));
    imageData = tf.image.resizeNearestNeighbor(imageData, [58, 58]);

    let modelInput = tf.image.resizeNearestNeighbor(imageData, [128, 128]);
    let result = classifier.predict(tf.reshape(modelInput, [1, 128, 128, 3]));

    let top = tf.topk(result, 2, true).indices.dataSync();

    let prediction = result.dataSync();
    let type;

    if (prediction[top[0]] / prediction[top[1]] > 2) {
      type = `${pokemonClasses[top[0]]}, ${pokemonClasses[top[1]]}`;
    } else {
      type = `${pokemonClasses[top[0]]}`;
    }

    $(`#${i}`).attr("data-original-title", `${type}`);
    tf.browser.toPixels(imageData, document.getElementById(`${i}`));
  }

  $('[data-toggle="tooltip"]').tooltip()
}

$('#actionButton').click(function() {
  window.requestAnimationFrame( () => {
      $("#generatePokemon").toast("show");
    } );
  
  generatePokemon();
})