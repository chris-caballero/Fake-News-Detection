function classifyText() {
    var inputText = document.getElementById("text-area-1").value;
    
    if (inputText.length === 0) {
      document.getElementById("text-required").style.display = 'block';
      return;
    } else {
      document.getElementById("text-required").style.display = 'none';
    }

    
    console.log('running classify text.');
    // Send a POST request to the server with the text data
    fetch('/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: 'text=' + encodeURIComponent(inputText)
    })
    .then(function(response) {
      return response.json();
    })
    .then(function(data) {
      var resultElement = document.getElementById("result");
      console.log(data.result);
      if (data.result < 35) {
        resultElement.innerHTML = 'Your text is likely to be Fake News.'
      } else if (data.result < 55) {
        resultElement.innerHTML = 'Your text is suspicious.'
      } else {
        resultElement.innerHTML = 'Your text is unlikely to be Fake News.'
      }
    })
    .catch(function(error) {
      console.log('Error:', error);
    });
}

function selectModel(button) {
  var button2;
  var model_type = button.value;
  button.classList.add("selected");

  if (button.value === 'uncased') {
    button2 = document.getElementById('cased')
  } else {
    button2 = document.getElementById('uncased')
  }

  if (button2.classList.contains("selected")) {
    button2.classList.remove("selected")
  }

  console.log('choosing new model:', model_type);
  // Send a POST request to the server with the text data
  fetch('/select_model', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: 'text=' + encodeURIComponent(model_type)
  })
  .then(function(response) {
    return response.json();
  })
  .then(function(data) {
    console.log(data.model)
  })
  .catch(function(error) {
    console.log('Error:', error);
  });
}

function updateCharacterCount() {
  var text_length = document.getElementById("text-area-1").value.length;
  var counter = document.getElementById("word-counter");
  counter.textContent= text_length + "/" + 250 + " characters";

}

updateCharacterCount();