import React from 'react'
import Typography from '@material-ui/core/Typography'
import {makeStyles} from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
    footer: {
        padding: theme.spacing(2)
    }
}));

const Footer = () => {
    const classes = useStyles();
    return (
        <footer className={classes.footer}>
            <Typography
                variant="subtitle1"
                align="center"
                color="textSecondary"
                component="p">
                Created by Wojciech Korczyński 2020
            </Typography>
        </footer>
    )
}
export default Footer;
