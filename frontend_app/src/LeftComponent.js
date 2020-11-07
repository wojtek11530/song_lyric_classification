import React from 'react'
import { TextField } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';

const useStyles = makeStyles(theme => ({
  container: {
    flexGrow: 1,
    textAlign: 'center',
    padding: theme.spacing(8, 0, 8)
  },
}));

const LeftComponent = () => {
    const classes = useStyles();
    return(
      <Container className={classes.container}>
           <TextField
              multiline
              rows={25}
              fullWidth
              variant="outlined"
            />
      </Container>
    )
}
export default LeftComponent;
