import React, {useContext} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    LabelList,
    ResponsiveContainer
} from 'recharts';
import Typography from '@material-ui/core/Typography'
import {Context} from "../Context";

const renderCustomAxisTick = ({x, y, payload}) => {
    let imageSrc = '';
    switch (payload.value) {
        case 'angry':
            imageSrc = 'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/google/263' +
                    '/angry-face_1f620.png';
            break;
        case 'happy':
            imageSrc = 'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/google/263' +
                    '/grinning-face_1f600.png';
            break;
        case 'sad':
            imageSrc = 'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/google/263' +
                    '/sad-but-relieved-face_1f625.png';
            break;
        case 'relaxed':
            imageSrc = 'https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/160/google/263' +
                    '/relieved-face_1f60c.png';
            break;
        default:
            imageSrc = '';
    }
    return (
        <g transform={`translate(${x},${y})`}>
            <text x={0} y={0} dy={14} textAnchor="middle" fill="#666" fontSize="1.2rem">{payload.value}</text>
            <image href={imageSrc} x={-24} y={18} width={48} height={48} fill="#666"/>
        </g>
    );
};

export default function ResultChart() {
    const theme = useTheme();

    const labelFormatter = (value) => {
        return Math.round(100 * value) + '%';
    };

    const {title, artist, lyrics, results, lyricsError} = useContext(Context);
    const [stateResults, setResults] = results;

    return (
        <React.Fragment>
            <Typography component="h2" variant="h5" color="primary" gutterBottom>
                Emotions Probabilities
            </Typography>
            <ResponsiveContainer>
                <BarChart
                    data={stateResults}
                    margin={{top: 16, right: 8, bottom: 82, left: 0}}>
                    <XAxis dataKey="mood" tick={renderCustomAxisTick}/>
                    <YAxis tick={{fontSize: '1.1rem'}} tickFormatter={labelFormatter}/>
                    <Bar dataKey="prob" barSize={90} fill={theme.palette.secondary.light}>
                        <LabelList
                            dataKey="prob"
                            position="top"
                            formatter={labelFormatter}
                            style={{fontSize: '1.2rem', fill: theme.palette.primary.dark, fontWeight: 500 }}
                        />
                    </ Bar>
                </BarChart>
            </ResponsiveContainer>
        </React.Fragment>
    );
}
