import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import Container from '@material-ui/core/Container';
import LeftComponent from './LeftComponent';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  paper: {
    padding: theme.spacing(2),
    textAlign: 'center',
    color: theme.palette.text.secondary,
    height: '80vh'
  },
}));

export default function Body() {
  const classes = useStyles();

  return (
    <Container maxWidth="lg" className={classes.root}>
        <Grid container spacing={3}>
            <Grid item md={6} xs={12}>
                <LeftComponent />
            </Grid>
            <Grid item md={6} xs={12}>
                <Paper className={classes.paper}>xs=6</Paper>
            </Grid>
        </Grid>
    </Container>
  );
}
