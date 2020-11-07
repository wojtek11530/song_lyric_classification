import React from 'react'
import { TextField } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Button from '@material-ui/core/Button';

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
  }


}));

const RightComponent = () => {
    const classes = useStyles();
    return(
      <Container className={classes.container}>
            <Button size="large"  variant="contained"  color="primary" disableElevation className={classes.button}>
              Get Mood!
            </Button>
      </Container>
    )
}
export default RightComponent;
