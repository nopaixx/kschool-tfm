import React, { Component } from 'react';
import logo from './logo.svg';
import angelimage from './20190424_194443.jpg';
import headerimage from './initial.png';
import ImageUpload from './uploadimage.js'
import './App.css';

class App extends Component {
  render() {
    return (
      <div className="App">
        <div className="App-header">
          <img src={headerimage} className="App-header2" alt="logo" />
          <h2>Welcome to Final TFM Datascience</h2>
        </div>
        <p className="App-intro">
          <img src={angelimage} className="App-angel_image" alt="angel image" />
          <h2>CarDrawing brushtool using DeepLearning</h2>
        </p>
        <p className="App-intro">
          My Name: Angel Lordan Roca          
        </p>
        <p className="App-intro">
          If you want to hire me please contact at:
        </p>
        <p className="App-intro">
          Phone: +34 650 735 499
        </p>
        <p className="App-intro">
          LinkedIn: https://www.linkedin.com/in/angel-lordan-3b854616b/
        </p>
        <div className="App-mask">
          <h2>First Model Trained-Masking car</h2>
          <h3>Please upload your favourite car</h3>

          h3 React Image Preview & Upload Component          
          div#mainApp
          div.centerText
          span Checkout associated 
          a(href="http://www.hartzis.me/react-image-upload/" target="_blank") blog post

        </div>  
            
      </div>
    );
  }
}

export default App;
