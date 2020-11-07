import React, { useState, useEffect } from 'react'
import { TextField } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography'

const useStyles = makeStyles(theme => ({
  container: {
    flexGrow: 1,
    textAlign: 'center',
    paddingTop: theme.spacing(1),
    paddingLeft: theme.spacing(0),
    paddingRight: theme.spacing(0)
  },

  textField: {
    backgroundColor: "white"
  },

  header: {
    flexGrow: 1,
    padding: theme.spacing(1, 1, 0),
    textAlign: 'left',
  }
}));

//const MyComponent = ({ location, ...otherProps }) => (whatever you want to render)

const LeftComponent = () => {
    const classes = useStyles();
    const divider = 20;
    const minRow = 12;

    const calcRows = () => {
        return Math.max(minRow, Math.round((window.innerHeight-200) / divider));
    }
    const [rows, setRows] = useState(calcRows());


    useEffect(() => {
       function handleResize() {
            setRows(calcRows())
        }

        window.addEventListener('resize', handleResize)
    });

    return(
      <Container className={classes.container}>
            <Typography variant="h6" className={classes.header}>
                {"Lyrics of song"}
            </Typography>
           <TextField className={classes.textField}
              multiline
              rows={rows}
              fullWidth
              variant="outlined"
            />
      </Container>
    )
}
export default LeftComponent;
