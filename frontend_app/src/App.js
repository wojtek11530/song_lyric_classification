import React, { Component } from 'react';
import CssBaseline from '@material-ui/core/CssBaseline';
import NavBar from './NavBar';
import Body from './Body';

class App extends Component {
  render() {
    return (
      <>
        <CssBaseline />
        <NavBar />
        <Body />
      </>
    )
  }
}
export default App