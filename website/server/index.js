const express = require('express');
const app = express();
const cors = require('cors');
const mysql = require('mysql2');

app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(express.json());


const db = mysql.createConnection({
    host: "localhost",
    user: "tyler",
    password: "root",
    database: "SeniorProject"
});

const con = db.connect((err) => {
    if (err) throw err;
    console.log("Connected successfully")
});

app.get('/', (req , res) => {
    res.send("Hello")
})

app.post('/rosters', (req, res) => {

    const teamId = req.body.teamId;
    const seasonId = req.body.seasonId;

    const sqlSelect = "SELECT Name, Pos, Num, G, GS, Age, College, Rosters.PId " +
    "FROM Rosters JOIN Players " +
    "ON Rosters.PId = Players.PId " +
    "WHERE Rosters.TId = ? and Rosters.SId = ?";
   
    db.query(sqlSelect, [teamId, seasonId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/passing', (req, res) => {

    const playerId = req.body.playerId;

    const sqlSelect = "SELECT SId, TId, Att, Cmp, Yds, YPA, YPG, TD, Ints, Sack, QBR " +
    "FROM PlayerPassing " +
    "WHERE PId = ?";
   
    db.query(sqlSelect, [playerId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/rushing', (req, res) => {

    const playerId = req.body.playerId;

    const sqlSelect = "SELECT SId, TId, Att, Yds, Avg, YPG, Lg, TD, `10+` " +
    "FROM PlayerRushing " +
    "WHERE PId = ?";
   
    db.query(sqlSelect, [playerId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/receiving', (req, res) => {

    const playerId = req.body.playerId;

    const sqlSelect = "SELECT SId, TId, Rec, Yds, Avg, YPG, Lg, TD, `20+`, Tar, YAC " +
    "FROM PlayerReceiving " +
    "WHERE PId = ?";
   
    db.query(sqlSelect, [playerId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/defense', (req, res) => {

    const playerId = req.body.playerId;

    const sqlSelect = "SELECT SId, TId, Ints, IntYds, IntAvg, IntLong, IntTD, Solo, Ast, Tot, Sack, SackYds " +
    "FROM PlayerDefense " +
    "WHERE PId = ?";
   
    db.query(sqlSelect, [playerId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/kicking', (req, res) => {

    const playerId = req.body.playerId;

    const sqlSelect = "SELECT SId, TId, FGM, FGA, FPct, `0-19`, `20-29`, `30-39`, `40-49`, `50+`, Lg, XPM, XPA, XPct, Pts " +
    "FROM PlayerKicking " +
    "WHERE PId = ?";
   
    db.query(sqlSelect, [playerId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/punting', (req, res) => {

    const playerId = req.body.playerId;

    const sqlSelect = "SELECT SId, TId, Num, Yds, Avg, Lg, TB, `In20`, `50+`, Blk " +
    "FROM PlayerPunting " +
    "WHERE PId = ?";
   
    db.query(sqlSelect, [playerId], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.post('/schedules', (req, res) => {

    const TId = req.body.team;
    const startDate = req.body.startDate;
    const endDate = req.body.endDate;
    console.log(startDate)
    console.log(endDate)

    const sqlSelect = "SELECT Date, Home, Away, H_Q1, H_Q2, H_Q3, H_Q4, A_Q1, A_Q2, A_Q3, A_Q4, H_Final, A_Final " +
    "FROM Schedules " +
    "WHERE TId = ? and Date >= ? and Date <= ?";
   
    db.query(sqlSelect, [TId, startDate, endDate], (err, result) => {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
})

app.listen(3001, () => {
    console.log("running on port 3001");
})