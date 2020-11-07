import React, { Component } from 'react';
import CssBaseline from '@material-ui/core/CssBaseline';
import Container from '@material-ui/core/Container';
import NavBar from './NavBar';
import Body from './Body';
import Footer from './Footer';
import { makeStyles } from '@material-ui/core/styles';


const useStyles = makeStyles(theme => ({
  main: {
    flex: 1
  },
  app: {
      display: "flex",
      minHeight: "100vh",
      flexDirection: "column"
  }
}));

const App = () => {
    const classes = useStyles();
    return (
      <div className={classes.app}>
        <CssBaseline />
        <NavBar />
        <main className={classes.main}>
        <Body />
        </main>
        < Footer/>
      </ div>
    )
}
export default App