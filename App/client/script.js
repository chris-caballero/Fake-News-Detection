function classifyText() {
    var inputText = document.getElementById("textInput").value;
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
      if (data.result < 35) {
        resultElement.innerHTML = 'Your text is likely Fake News'
      } else if (data.result < 60) {
        resultElement.innerHTML = 'Your text is suspicious'
      } else {
        resultElement.innerHTML = 'Your text is unlikely to be Fake News'
      }
    })
    .catch(function(error) {
      console.log('Error:', error);
    });
  }
  