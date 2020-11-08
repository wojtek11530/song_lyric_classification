import React from 'react'
import AppBar from '@material-ui/core/AppBar'
import Toolbar from '@material-ui/core/Toolbar'
import Typography from '@material-ui/core/Typography'
import { makeStyles } from '@material-ui/core/styles';
import MusicNoteIcon from '@material-ui/icons/MusicNote';

const useStyles = makeStyles(theme => ({
  toolbar: {
    justifyContent: 'center',
  },

  title: {
    textAlign: 'center',
    fontFamily: 'Rambla, sans-serif'
  },
}));

const NavBar = () => {
    const classes = useStyles();
    return(
        <AppBar position="relative">
            <Toolbar className={classes.toolbar}>
                <MusicNoteIcon fontSize="large" />
                <Typography variant="h4" className={classes.title}>
                    Lyrics Emotions
                </Typography>
                <MusicNoteIcon fontSize="large" />
            </Toolbar>
        </AppBar>
    )
}
export default NavBar;
