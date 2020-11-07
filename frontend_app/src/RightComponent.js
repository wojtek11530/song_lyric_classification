import React, {useState, useContext} from 'react'
import { TextField } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import ResultChart from './ResultChart';
import { Context } from "./Context";

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
      margin: theme.spacing(10, 1, 1, 1),
    },
    fontSize: '1.7rem',
    textTransform: 'none'
  },

  paper: {
    backgroundColor: 'white',
    padding: theme.spacing(1),
    display: 'flex',
    overflow: 'auto',
    flexDirection: 'column',
    height: 280,
    margin: theme.spacing(1, 1, 1, 1),
    [theme.breakpoints.up('sm')]: {
      margin: theme.spacing(4, 2, 1, 2),
    },
    [theme.breakpoints.up('md')]: {
      margin: theme.spacing(4, 4, 1, 4),
    },
  }
}));


function createData(mood, prob) {
  return { mood, prob };
}

const data = [
  createData('angry', 0.12),
  createData('happy', 0.18),
  createData('relaxed', 0.60),
  createData('sad', 0.10)
];

const RightComponent = () => {
    const classes = useStyles();

    const { lyrics, results, lyricsError } = useContext(Context);
    const [stateLyrics, setLyrics] = lyrics;
    const [stateResults, setResults] = results;
    const [stateLyricsError, setLyricsError] = lyricsError;

    const [showResults, setShowResults] = useState(false);

    const [buttonDisabled, setButtonDisabled] = useState(false);

    const onButtonClick =  () => {
        setButtonDisabled(true);
        if (stateLyrics == '') {
            setShowResults(false);
            setLyricsError(true);
        } else {
            setLyricsError(false);
            setResults(data);
            setShowResults(true);
        }
        setButtonDisabled(false);
    }

    return(
      <Container className={classes.container}>
            <Button size="large"  variant="contained"  color="primary" disableElevation className={classes.button}
                disabled={buttonDisabled} onClick={onButtonClick}>
                Get Mood!
            </Button>
            { showResults ?
            <Paper className={classes.paper}>
                <ResultChart />
            </Paper>
            : null }

      </Container>
    )
}
export default RightComponent;
