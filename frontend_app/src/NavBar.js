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
        <div>
        <AppBar position="static">
            <Toolbar>
                <Typography variant="h5" className={classes.title}>
                    Lyric Mood
                </Typography>
            </Toolbar>
        </AppBar>
        </div>
    )
}
export default NavBar;
