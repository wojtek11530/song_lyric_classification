import React, {useState, useContext} from 'react';
import axios from 'axios';
import {makeStyles} from '@material-ui/core/styles';
import Alert from '@material-ui/lab/Alert'
import Container from '@material-ui/core/Container';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import Box from '@material-ui/core/Box';
import ResultChart from './ResultChart';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import {Context} from "../Context";

const useStyles = makeStyles(theme => ({
    container: {
        flexGrow: 1,
        textAlign: 'center',
        paddingLeft: theme.spacing(0),
        paddingRight: theme.spacing(0)
    },

    box: {
     margin: theme.spacing(1, 1, 1, 1),
        [theme.breakpoints.up('sm')]: {
            margin: theme.spacing(8, 1, 4, 1)
        },
    },

    button: {
        fontSize: '1.7rem',
        padding: theme.spacing(2, 3),
        fontWeight: 400,
        lineHeight: 1,
        textTransform: 'none',
    },

    deleteButton: {
        margin: theme.spacing(1.5, 1.5),
        padding: theme.spacing(1.5, 1),
        fontSize: '0.85rem',
        lineHeight: 1,
        fontWeight: 400,
        textTransform: 'none'
    },

    smallButton: {
        margin: theme.spacing(1,0),
        fontSize: '1.2rem',
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
            margin: theme.spacing(1, 2, 1, 2)
        },
        [theme.breakpoints.up('md')]: {
            margin: theme.spacing(1, 4, 1, 4)
        }
    }
}));

function createData(mood, prob) {
    return {mood, prob};
}

const RightComponent = () => {
    const classes = useStyles();

    const {title, artist, lyrics, lyricsError, showResults, showAverageResults} = useContext(Context);
    const [stateTitle, setTitle] = title;
    const [stateArtist, setArtist] = artist;
    const [stateLyrics, setLyrics] = lyrics;
    const [stateLyricsError, setLyricsError] = lyricsError;
    const [stateShowResults, setShowResults] = showResults;
    const [stateShowAverageResults, setShowAverageResults] = showAverageResults;

    const [buttonDisabled, setButtonDisabled] = useState(false);
    const [showErrorMessage, setShowErrorMessage] = useState(false);
    const [averageResultButtonName, setAverageResultButtonName] = useState('Show average songs emotions');
    const [results, setResults] = useState([]);
    const [averageResults, setAverageResults] = useState([]);
    const [showDeleteButton, setShowDeleteButton] = useState(false);

    const [openDialog, setOpenDialog] = useState(false);
    const [dialogMessage, setDialogMessage] = useState('Removing of song emotions results has succeeded');

    axios.get('http://localhost:5000/results_count')
        .then(response => {
            console.log(response);
            if (response.status === 200) {
                let count = response.data['count']
                if (count > 0) {
                    setShowDeleteButton(true);
                } else {
                    setShowDeleteButton(false);
                }
            }
        })
        .catch(error => console.log(error));

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
                    setShowDeleteButton(true);
                    setShowErrorMessage(false);
                    setShowResults(true);
                    setShowAverageResults(stateShowAverageResults);
                }
            })
            .catch(error => console.log(error));
    }

    const onButtonClick = () => {
        setButtonDisabled(true);
        if (stateLyrics === '') {
            setShowResults(false);
            setShowAverageResults(false);
            setLyricsError(true);
            setShowErrorMessage(false);
        } else {
            setLyricsError(false);
            fetchEmotionResults();
        }
        setButtonDisabled(false);
    }

    const onSmallButtonClick = () => {
        if (stateShowAverageResults == true) {
            setAverageResultButtonName('Show average songs emotions');
        } else {
            setAverageResultButtonName('Hide average songs emotions');
        }
        setShowAverageResults(!stateShowAverageResults);
    }

    const handleClose = () => {
        setOpenDialog(false);
    };

    const onDeleteButtonClick = () => {
        axios
            .get('http://localhost:5000/delete_emotions')
            .then(response => {
                if (response.status === 200) {
                    setShowResults(false);
                    setShowAverageResults(false);
                    setShowDeleteButton(false);
                    setOpenDialog(true);
                    setAverageResultButtonName('Show average songs emotions');
                    setDialogMessage('Removing of song emotions results has succeeded');
                } else {
                    setOpenDialog(true);
                    setDialogMessage('Removing of song emotions results has failed');
                }
            })
            .catch(error => console.log(error));
    }

    return (
        <>
        <Container className={classes.container}>
            <Box display="flex" flexDirection="row" justifyContent="center" className={classes.box}>
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
                {showDeleteButton
                ? <>
                    <Button
                        size="small"
                        variant="contained"
                        disableElevation
                        className={classes.deleteButton}
                        onClick={onDeleteButtonClick}>
                        {'Delete saved emotion results'}
                    </Button>
                  </>
                : null}
            </Box>
            {showErrorMessage
                ?  <Alert severity="error" className={classes.alert}>
                        The emotion prediction has failed. The lyrics might be too short.
                   </Alert>
                : null}
            {stateShowResults
                ? <>
                    <Paper className={classes.paper}>
                        <ResultChart title={'Song Emotion Probabilities'} stateResults={results}/>
                    </Paper>
                     <Button
                        size="medium"
                        variant="contained"
                        color="primary"
                        disableElevation
                        className={classes.smallButton}
                        onClick={onSmallButtonClick}>
                        {averageResultButtonName}
                    </Button>
                  </>
                : null}
            {stateShowAverageResults
                ?
                    <Paper className={classes.paper}>
                        <ResultChart title={'Average Emotion Probabilities'} stateResults={averageResults}/>
                    </Paper>
                : null}
        </Container>
        <Dialog open={openDialog} onClose={handleClose} aria-labelledby="alert-dialog-title"
            aria-describedby="alert-dialog-description">
            <DialogTitle id="alert-dialog-title">
                {"Song emotion results removing"}
            </DialogTitle>
            <DialogContent>
                <DialogContentText id="alert-dialog-description">
                    {dialogMessage}
                </DialogContentText>
            </DialogContent>
            <DialogActions>
                <Button onClick={handleClose} color="primary">
                    Ok
                </Button>
            </DialogActions>
        </Dialog>
        </>
    )
}
export default RightComponent;
