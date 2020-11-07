import React from 'react';
import { useTheme } from '@material-ui/core/styles';
import { BarChart, Bar, XAxis, YAxis, Label, LabelList, ResponsiveContainer } from 'recharts';
import Typography from '@material-ui/core/Typography'

function createData(mood, prob) {
  return { mood, prob };
}

const data = [
  createData('angry', 0.12),
  createData('happy', 0.18),
  createData('relaxed', 0.60),
  createData('sad', 0.10)
];

export default function ResultChart() {
  const theme = useTheme();
  const labelFormatter = (value) => {
    return Math.round(100*value) + '%';
  };
  return (
    <React.Fragment>
    <Typography component="h2" variant="h6" color="primary" gutterBottom>
      Mood Probabilities
    </Typography>
    <ResponsiveContainer>
        <BarChart data={data} margin={{top: 20, right: 8, bottom: 0, left: 0}}>
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