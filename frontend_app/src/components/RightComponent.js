import React, {useState, useContext} from 'react';
import axios from 'axios';
import {makeStyles} from '@material-ui/core/styles';
import Alert from '@material-ui/lab/Alert'
import Container from '@material-ui/core/Container';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import ResultChart from './ResultChart';
import {Context} from "../Context";

const useStyles = makeStyles(theme => ({
    container: {
        flexGrow: 1,
        textAlign: 'center',
        paddingLeft: theme.spacing(0),
        paddingRight: theme.spacing(0)
    },

    button: {
        margin: theme.spacing(1, 1, 1, 1),
        [theme.breakpoints.up('sm')]: {
            margin: theme.spacing(10, 1, 1, 1)
        },
        fontSize: '1.7rem',
        fontWeight: 400,
        textTransform: 'none'
    },

    alert: {
        margin: theme.spacing(1, 0, 1, 0),
        [theme.breakpoints.up('sm')]: {
            margin: theme.spacing(2, 1, 0, 1)
        },
        [theme.breakpoints.up('md')]: {
            margin: theme.spacing(2, 4, 0, 4)
        },
    },

    paper: {
        backgroundColor: 'white',
        padding: theme.spacing(1),
        height: 350,
        margin: theme.spacing(1, 1, 1, 1),
        [theme.breakpoints.up('sm')]: {
            margin: theme.spacing(4, 2, 1, 2)
        },
        [theme.breakpoints.up('md')]: {
            margin: theme.spacing(4, 4, 1, 4)
        }
    }
}));

function createData(mood, prob) {
    return {mood, prob};
}

const RightComponent = () => {
    const classes = useStyles();

    const {title, artist, lyrics, lyricsError} = useContext(Context);
    const [stateTitle, setTitle] = title;
    const [stateArtist, setArtist] = artist;
    const [stateLyrics, setLyrics] = lyrics;
    const [stateLyricsError, setLyricsError] = lyricsError;

    const [showErrorMessage, setShowErrorMessage] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [results, setResults] = useState([]);
    const [averageResults, setAverageResults] = useState([]);
    const [buttonDisabled, setButtonDisabled] = useState(false);

    const fetchEmotionResults = () => {
        axios
            .post('http://localhost:5000/song_emotion', {lyrics: stateLyrics, title: stateTitle, artist: stateArtist})
            .then(response => {
                console.log(response);
                if (response.status === 204) {
                    setResults(null);
                    setShowResults(false);
                    setShowErrorMessage(true);
                } else {
                    let formattedData = [];
                    for (const property in response.data[0]) {
                        formattedData.push(createData(property, response.data[0][property]));
                    }
                    setResults(formattedData);

                    let formattedAverageData = [];
                    for (const property in response.data[1]) {
                        formattedAverageData.push(createData(property, response.data[1][property]));
                    }
                    setAverageResults(formattedAverageData);

                    setShowErrorMessage(false);
                    setShowResults(true);
                }
            })
            .catch(error => console.log(error));
    }

    const onButtonClick = () => {
        setButtonDisabled(true);
        if (stateLyrics === '') {
            setShowResults(false);
            setLyricsError(true);
            setShowErrorMessage(false);
        } else {
            setLyricsError(false);
            fetchEmotionResults();
        }
        setButtonDisabled(false);
    }

    return (
        <Container className={classes.container}>
            <Button
                size="large"
                variant="contained"
                color="primary"
                disableElevation
                className={classes.button}
                disabled={buttonDisabled}
                onClick={onButtonClick}>
                Get Emotions!
            </Button>
            {showErrorMessage
                ?  <Alert severity="error" className={classes.alert}>
                        The emotion prediction has failed. The lyrics might be too short.
                   </Alert>
                : null}
            {showResults
                ? <>
                    <Paper className={classes.paper}>
                        <ResultChart title={'Song Emotion Probabilities'} stateResults={results}/>
                    </Paper>
                    <Paper className={classes.paper}>
                        <ResultChart title={'Average Emotion Probabilities'} stateResults={averageResults}/>
                    </Paper>
                  </>
                : null}
        </Container>
    )
}
export default RightComponent;
