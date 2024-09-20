/*
*
* DATA related functions
*/

// Samples 20 means, 10 for each class for later data generation
var sample_means = function() {
    var identidy_mat = [
        [ 2.0, 0.0,],
        [ 0.0, 2.0]
    ];

    var mean_sample_size = 10;

    var mean_distribution_class1 = MultivariateNormal([0.5,-0.5], identidy_mat)
    var mean_distribution_class2 = MultivariateNormal([-0.5,0.5], identidy_mat)

    var means_class_1 = [];
    var means_class_2 = [];
    
    for(var i = 0; i < mean_sample_size; i++)
    {   
        var mean_1 = mean_distribution_class1.sample()
        means_class_1.push({
            'x': mean_1[0],
            'y': mean_1[1],
            'label': 1
        });

        var mean_2 = mean_distribution_class2.sample();
        means_class_2.push({
            'x': mean_2[0],
            'y': mean_2[1],
            'label': -1
        });
    }

    return [d3.shuffle(means_class_1), d3.shuffle(means_class_2)]
}

//Samples sampleSize datapoints from the means vector
var generate_mixture_from_means = function(sampleSize, noise, means)
{
    var means_class_1 = means[0]
    var means_class_2 = means[1]

    var data = []
    var identidy_mat_small = [
        [noise, 0.0,],
        [ 0.0, noise]
    ];
    for(var i = 0; i < sampleSize/2; i++)
    {   
        var idx = Math.floor(Math.random()*means_class_1.length)
        var mean_class_1 = means_class_1[idx];
        var distribution_class_1 = MultivariateNormal([mean_class_1.x, mean_class_1.y], identidy_mat_small)
        var sample_point = distribution_class_1.sample()
        data.push({
            'x': sample_point[0],
            'y': sample_point[1],
            'label': 1,
            'mean': idx
        })

        var idx2 = Math.floor(Math.random()*means_class_2.length);
        var mean_class_2 = means_class_2[idx2];
        var distribution_class_2 = MultivariateNormal([mean_class_2.x, mean_class_2.y], identidy_mat_small)
        sample_point = distribution_class_2.sample()
        data.push({
            'x': sample_point[0],
            'y': sample_point[1],
            'label': -1,
            'mean': idx2
        })

    }

    return data

}


//Construct KD-tree for data 
var constructKDTree = function(points) {
    
    //Euclidean distance betweeen two points without the square root for optimization
    //since we only need relative distances 
    var distance = function(a, b){
        return Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2);
    }
    
    var tree = new kdTree(points, distance, ["x", "y"]);
    return tree;
}

var getDartBoardPoints = function(bias, variance) {
    
    var theta = Math.floor(Math.random() * (360 - 0 + 1)) + 0;
    var xCoord = bias*math.cos(math.unit(theta, 'deg'));
    var yCoord = bias*math.sin(math.unit(theta, 'deg'));
    var points = []
    var n = 20;

    var identidy_mat = [
        [ variance, 0.0,],
        [ 0.0, variance]
    ];

    var distribution = MultivariateNormal([xCoord, yCoord], identidy_mat)
    
    for(var j = 0; j < 10; j++) {
        var point = distribution.sample();
        points.push({
                'x': point[0],
                'y': point[1]
            });
        }

    return points
    
}

var getVarianceOfModel = function(noise, k)
{

    return noise/k;
}


var getBiasOfModel = function(means, k)
{
    var num_training_sets = 100;

    var errors = []
    var meanvector = means[0].concat(means[1])
    for(var i = 0; i < num_training_sets; i++)
    {
        var data = generate_mixture_from_means(400,noiseMap(noise), means);
    
        // filter out points that fall out of our domain
        data = data.filter(p => {
            return p.x >= domain[0] && p.x <= domain[1]
                && p.y >= domain[0] && p.y <= domain[1];
            });
        
        var tree = constructKDTree(data);

        
        var error = calculate_error(tree, meanvector, kval);
        errors.push(error)
        
    
    
    }

    return errors
    

}

var calculate_error = function(tree,dat, kval) {
    
    var error = 0;
    for(var i = 0; i<dat.length; i++)
    {
        var nearest = tree.nearest({ x: dat[i].x, y: dat[i].y }, kval);
            
            var sum = 0;
            var votes = []
            for (var h = 0; h < nearest.length; h++)
            {   
                votes.push(nearest[h][0].label)
                sum = sum + (nearest[h][0].label == -1 ? -1 : 1);
                
            }

            var predicted_label = sum > 0 ? 1 : -1;
            //console.log("Predicted/True label: {0}, {1} ".format(predicted_label, dat[i].label))
            var error = error + (predicted_label == dat[i].label ? 0 : 1);
            
            // console.log(error)
    }

    return error/dat.length;
}