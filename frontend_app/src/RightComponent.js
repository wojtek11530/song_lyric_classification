import React from 'react'
import { TextField } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';

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
    backgroundColor: '#cfe8fc',
    height: '40vh',
    margin: theme.spacing(1, 1, 1, 1),
    [theme.breakpoints.up('sm')]: {
      margin: theme.spacing(4, 2, 1, 2),
    },
    [theme.breakpoints.up('md')]: {
      margin: theme.spacing(4, 4, 1, 4),
    },
  }


}));

const RightComponent = () => {
    const classes = useStyles();
    return(
      <Container className={classes.container}>
            <Button size="large"  variant="contained"  color="primary" disableElevation className={classes.button}>
                Get Mood!
            </Button>
            <Paper className={classes.paper}/>
      </Container>
    )
}
export default RightComponent;
