/*

        DARTBOARD SETUP

*/

const dartBoardSize = 250; //dartboard size in px size*size
var svgDartBoard = d3.select("#dart").append("svg")
var dartdomain = [-3.5,3.5]; //the actual coordinate boundaries that we will use

svgDartBoard.attr("width", dartBoardSize)
.attr("height", dartBoardSize)
.append("g")



// Set up our scales
var dartScale = d3.scale.linear()
.domain(dartdomain)
.range([0, dartBoardSize]);

var dartRadiusScale = d3.scale.linear()
.domain([0,2])
.range([0, dartBoardSize/2]); 



var noiseMap = d3.scale.linear().domain([10,100]).range([0.05,0.2])
var biasMap = d3.scale.linear().domain([0,0.3]).range([0,3])




var clearDartBoard = function() {
    svgDartBoard.selectAll(".train").remove();
}

var renderDartBoard = function(points) {

    //Clear old points
    svgDartBoard.selectAll(".train").remove();

    var directions = [-4,4];
    //Draw new ones
    svgDartBoard.selectAll(".dot")
        .data(points)
        .enter().append("circle")
      .attr("class", "dot")
      .attr("class", "train")
      .attr("r", 3.5)
      .attr("cx", function() {
          var rand = xScale(directions[Math.floor(Math.random() * directions.length)])
          return rand
        })
      .attr("cy", function() {
          return  yScale(directions[Math.floor(Math.random() * directions.length)])
        })
      .style("fill", "blue")
      .transition()
        .delay(function(d,i) {
            return i*200;
        })
        .attr("cx", function (d) { return dartScale(d.x); })
        .attr("cy", function (d) { return dartScale(d.y); })
        .duration(1000)
        .ease("linear")


    

}
