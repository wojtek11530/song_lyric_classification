import React from 'react'
import AppBar from '@material-ui/core/AppBar'
import Toolbar from '@material-ui/core/Toolbar'
import Typography from '@material-ui/core/Typography'
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
  title: {
    flexGrow: 1,
    textAlign: 'center',
  },
}));

const NavBar = () => {
    const classes = useStyles();
    return(
        <AppBar position="relative">
            <Toolbar>
                <Typography variant="h5" className={classes.title}>
                    Lyrics Emotions
                </Typography>
            </Toolbar>
        </AppBar>
    )
}
export default NavBar;
