import React from 'react';
import {makeStyles} from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Container from '@material-ui/core/Container';
import LeftComponent from './LeftComponent';
import RightComponent from './RightComponent';

const useStyles = makeStyles((theme) => ({
    bodyContainer: {
        flexGrow: 1,
        paddingTop: theme.spacing(1),
        paddingBottom: theme.spacing(1)
    }
}));

export default function Body() {
    const classes = useStyles();

    return (
        <Container maxWidth="lg" className={classes.bodyContainer}>
            <Grid container spacing={1}>
                <Grid item md={6} sm={5} xs={12}>
                    <LeftComponent/>
                </Grid>
                <Grid item md={6} sm={7} xs={12}>
                    <RightComponent/>
                </Grid>
            </Grid>
        </Container>
    );
}
