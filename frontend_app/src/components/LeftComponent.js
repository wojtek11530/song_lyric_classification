import React, {useState, useEffect, useContext} from 'react'
import {TextField} from '@material-ui/core';
import {makeStyles} from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import {Context} from "../Context";

const useStyles = makeStyles(theme => ({
    container: {
        flexGrow: 1,
        textAlign: 'center',
        paddingTop: theme.spacing(1),
        paddingLeft: theme.spacing(0),
        paddingRight: theme.spacing(0)
    },

    helperText: {
        fontSize: '1.25rem'
    },

    input: {
        backgroundColor: 'white '
    },

    header: {
        flexGrow: 1,
        padding: theme.spacing(1, 1.3, 0),
        textAlign: 'left',
        fontStyle: 'italic'
    },
}));

const LeftComponent = () => {
    const classes = useStyles();
    const divider = 32;
    const minRow = 12;
    const calcRows = () => {
        return Math.max(minRow, Math.round((window.innerHeight - 200) / divider));
    }
    const [rows, setRows] = useState(calcRows());

    useEffect(() => {
        function handleResize() {
            setRows(calcRows())
        }
        window.addEventListener('resize', handleResize)
    });

    const {title, artist, lyrics, lyricsError, showResults, showAverageResults} = useContext(Context);
    const [stateTitle, setTitle] = title;
    const [stateArtist, setArtist] = artist;
    const [stateLyrics, setLyrics] = lyrics;
    const [stateLyricsError, setLyricsError] = lyricsError;

    return (
        <Container className={classes.container}>
            <Typography variant="h5" className={classes.header}>
                {"Song title"}
            </Typography>
            <TextField
                InputProps={{className: classes.input}}
                FormHelperTextProps={{className: classes.helperText}}
                fullWidth
                variant="outlined"
                onChange={(event) => setTitle(event.target.value)}
            />
            <Typography variant="h5" className={classes.header}>
                {"Artist"}
            </Typography>
            <TextField
                InputProps={{className: classes.input}}
                FormHelperTextProps={{className: classes.helperText}}
                fullWidth
                variant="outlined"
                onChange={(event) => setArtist(event.target.value)}
            />
            <Typography variant="h5" className={classes.header}>
                {"Song lyrics"}
            </Typography>
            <TextField
                InputProps={{className: classes.input}}
                FormHelperTextProps={{className: classes.helperText}}
                multiline
                rows={rows}
                fullWidth
                variant="outlined"
                error={stateLyricsError}
                helperText={stateLyricsError ? 'Empty field!' : ''}
                onChange={(event) => setLyrics(event.target.value)}
            />
        </Container>
    )
}
export default LeftComponent;
