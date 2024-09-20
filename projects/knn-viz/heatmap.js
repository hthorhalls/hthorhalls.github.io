/**
 * 
 *  HEATMAP 
 * 
 */


const DENSITY = 30; //logical width and height for grid
const NUM_SHADES = 30; // How detailed in color is decision boundary


// COLOR RANGE ON HEATMAP
// BORROWED FROM TENSORFLOW
var tmpScale = d3.scale.linear()
.domain([0, .5, 1])
.range(["#f59322", "#e8eaeb", "#0877bd"])
.clamp(true);

var colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(a => {
return tmpScale(a);
});

var color = d3.scale.quantize()
             .domain([-1, 1])
             .range(colors);



//Calculate decision boundary
var getDecisionBoundary = function(tree, kval) {
    
    
        var densityToCoordinateX = d3.scale.linear().domain([0, DENSITY-1]).range(domain);
        var densityToCoordinateY = d3.scale.linear().domain([DENSITY-1,0]).range(domain);
        
        //define a grid with the given density 
        var grid = new Array(DENSITY);
        for(k = 0; k < DENSITY; k++)
        {
            grid[k] = new Array(DENSITY);
        }
    
        for(i = 0; i < DENSITY; i++)
        {
            for(j = 0; j < DENSITY; j++) {
                
                var nearest = tree.nearest({ x: densityToCoordinateX(i), y: densityToCoordinateY(j) }, kval);
                
                var sum = 0;
                var votes = []
                for (var h = 0; h < nearest.length; h++)
                {   
                    votes.push(nearest[h][0].label)
                    sum = sum + (nearest[h][0].label == -1 ? -1 : 1);
                    
                }
                
                
                grid[i][j] = sum/kval
                
            }
        }
        return grid
}

//Render heatmap from updated grid values
var updateHeatmap = function(grid){
    
    
        var context = canvas.node().getContext('2d');
        var image = context.createImageData(DENSITY, DENSITY);
    
    
        for(j = 0, k = -1; j < DENSITY; ++j)
        {
            for(i = 0; i < DENSITY; ++i)
            {   
                c = d3.rgb(color(grid[i][j]));
                image.data[++k] = c.r;
                image.data[++k] = c.g;
                image.data[++k] = c.b;
                image.data[++k] = 190;
            }
        }
       
        context.putImageData(image, 0, 0);
    
    
    }