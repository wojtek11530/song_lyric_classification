import React, { useContext } from 'react';
import { useTheme } from '@material-ui/core/styles';
import { BarChart, Bar, XAxis, YAxis, LabelList, ResponsiveContainer } from 'recharts';
import Typography from '@material-ui/core/Typography'
import { Context } from "../Context";


export default function ResultChart() {
  const theme = useTheme();
  const labelFormatter = (value) => {
    return Math.round(100*value) + '%';
  };

  const { lyrics, results, lyricsError } = useContext(Context);
  const [stateResults, setResults] = results;

  return (
    <React.Fragment>
    <Typography component="h2" variant="h6" color="primary" gutterBottom>
        Emotions Probabilities
    </Typography>
    <ResponsiveContainer>
        <BarChart data={stateResults} margin={{top: 20, right: 8, bottom: 0, left: 0}}>
            <XAxis dataKey="mood" />
            <YAxis tickFormatter={labelFormatter}/>
            <Bar dataKey="prob" barSize={90} fill={theme.palette.secondary.light}>
                <LabelList dataKey="prob" position="top" formatter={labelFormatter}/>
            </ Bar>
        </BarChart>
    </ResponsiveContainer>
    </React.Fragment>
  );
}