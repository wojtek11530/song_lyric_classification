import React, {useState} from 'react';
import CssBaseline from '@material-ui/core/CssBaseline';
import {Context} from "./Context.js";
import NavBar from './components/NavBar';
import Body from './components/Body';
import Footer from './components/Footer';
import {makeStyles} from '@material-ui/core/styles';
import {ThemeProvider} from '@material-ui/core/styles';
import {createMuiTheme} from '@material-ui/core/styles'

const theme = createMuiTheme({

    breakpoints: {
        values: {
          xs: 0,
          sm: 720,
          md: 880,
          lg: 1280,
          xl: 1920,
        },
    },

    palette: {
        primary: {
            main: '#084887'
        },
        secondary: {
            main: '#F9AB55'
        }
    }
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

    const [title, setTitle] = useState('');
    const [artist, setArtist] = useState('');
    const [lyrics, setLyrics] = useState('');
    const [lyricsError, setLyricsError] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [showAverageResults, setShowAverageResults] = useState(false);

    const contextValues = {
        title: [title, setTitle],
        artist: [artist, setArtist],
        lyrics: [lyrics, setLyrics],
        lyricsError: [lyricsError, setLyricsError],
        showResults: [showResults, setShowResults],
        showAverageResults: [showAverageResults, setShowAverageResults],
    };

    return (
        <ThemeProvider theme={theme}>
            <Context.Provider value={contextValues}>
                <div className={classes.app}>
                    <CssBaseline/>
                    <NavBar/>
                    <main className={classes.main}>
                        <Body/>
                    </main>
                    < Footer/>
                </ div>
            </ Context.Provider>
        </ ThemeProvider>
    )
}
export default App
