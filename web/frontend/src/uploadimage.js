import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import './App.css';

class ImageUpload extends React.Component {
  constructor(props) {
    super(props);
    this.state = {file: '',imagePreviewUrl: '',finishedPhoto:''};
  }

  _handleSubmit(e) {
    e.preventDefault();
    // TODO: do something with -> this.state.file
    console.log('handle uploading-', this.state.file);
  }


  _handleImageChange(e) {
    e.preventDefault();

    let reader = new FileReader();
    let file = e.target.files[0];

    reader.onloadend = () => {
      this.setState({
        file: file,
        imagePreviewUrl: reader.result
      });
    }

    reader.readAsDataURL(file)
  }

  _handleConver = () => {
    this.setState({finishedPhoto: true})
  }
  _handleSubmit_applymask = () => {
    
    // TODO: do something with -> this.state.file
    
    this.setState({finishedMask: true})
  }

  render() {
    let {imagePreviewUrl, finishedPhoto, finishedMask} = this.state;
    let $imagePreview = null;
    let newPhoto = null;
    let applyMaskPhoto = null;
    if (imagePreviewUrl) {
      $imagePreview = (<img className={'imagen1'} src={imagePreviewUrl} />);
    } else {
      $imagePreview = (<div className="previewText">Please select an Image for Preview</div>);
    }
    if(finishedPhoto) {
      newPhoto = (<img className={'imagen1'} src={''} />);
    }else {
      newPhoto = (<div className="previewText">No Result yet</div>);
    }
    if(finishedMask){
      applyMaskPhoto = (<img className={'imagen1'} src={''} />);
    }else{
      applyMaskPhoto = (<div className="previewText">No Result yet</div>);
    }

    console.log(imagePreviewUrl)

    return (
      <div>
      <div className="previewComponent">
        <form onSubmit={(e)=>this._handleSubmit(e)}>
          <input className="fileInput" 
            type="file" 
            onChange={(e)=>this._handleImageChange(e)} />
          <button className="submitButton" 
            type="submit" 
            onClick={(e)=>this._handleSubmit(e)}>Upload Image</button>
        </form>
        <div className = {'imagenesDiv'}>
          <div className="imgPreview">
            {$imagePreview}
          </div>
          <button  onClick={(e)=>this._handleConver(e)}>
            hola
          </button>
          <div className="imgPreview">
            {newPhoto}
          </div>
        </div>
      </div>
      <div className="App-apply-mask">
        <h2>Apply mask to original Car</h2>
          <div className="imgPreview">
            {newPhoto}
          </div>
          <button  onClick={(e)=>this._handleSubmit_applymask(e)}>
            hola-2
          </button>
          <div className="imgPreview">
            {applyMaskPhoto}
          </div>
      </div>
      </div>
    )
  }
}
  
export default ImageUpload;
