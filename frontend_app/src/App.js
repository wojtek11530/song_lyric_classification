import React, { useState } from 'react';
import CssBaseline from '@material-ui/core/CssBaseline';
import { Context } from "./Context.js";
import NavBar from './components/NavBar';
import Body from './components/Body';
import Footer from './components/Footer';
import { makeStyles } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/core/styles';
import { createMuiTheme }  from '@material-ui/core/styles'


const theme = createMuiTheme({
  palette: {
    primary: {
        main: '#084887',
    },
    secondary: {
        main: '#F9AB55'
    },
  },
});

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

    const [lyrics, setLyrics] = useState('');
    const [results, setResults] = useState([]);
    const [lyricsError, setLyricsError] = useState(false);
    const contextValues = {
        lyrics: [lyrics, setLyrics],
        results: [results, setResults],
        lyricsError: [lyricsError, setLyricsError]
    };

    return (
        <ThemeProvider theme={theme}>
            <Context.Provider value={contextValues}>
              <div className={classes.app}>
                <CssBaseline />
                <NavBar />
                <main className={classes.main}>
                <Body />
                </main>
                < Footer/>
              </ div>
            </ Context.Provider>
        </ ThemeProvider>
    )
}
export default App
