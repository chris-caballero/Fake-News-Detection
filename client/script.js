function classifyText() {
    var inputText = document.getElementById("textInput").value;
  
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
      resultElement.innerHTML = "Classification result: " + data.result;
    })
    .catch(function(error) {
      console.log('Error:', error);
    });
  }
  