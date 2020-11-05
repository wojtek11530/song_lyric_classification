import React from 'react'
import { TextField } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';

const useStyles = makeStyles(theme => ({
  title: {
    flexGrow: 1,
    textAlign: 'center',
  },
}));

const LeftComponent = () => {
    const classes = useStyles();
    return(
      <Container>
            <TextField id="outlined-basic" label="Outlined" variant="outlined" />
      </Container>
    )
}
export default LeftComponent;
