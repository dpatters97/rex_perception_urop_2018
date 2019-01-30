#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <vector>
#include <array>

#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/io/pcd_io.h>
#include<pcl/surface/concave_hull.h>
#include<pcl/Vertices.h>
#include <pcl/io/hdl_grabber.h>


#include <pcl/filters/extract_indices.h>

using namespace std;
using namespace pcl;

// Point Type
// PointXYZ, PointXYZI, PointXYZRGBA
typedef PointXYZRGBA PointType;




namespace Filters
{
	const double VEHICLE_SIZE = 3.0; // size of box to filter around the vehicle
	const float HEIGHT_ABOVE_WATER = 5.0; // magintude of lower threshold (ie height of lidar above water)
	const float HEIGHT_ABOVE_LIDAR = 3.0; // magintude of upper threshold (ie difference in height between top of the vehicle and lidar. generous threshold used to ensure a sufficient amount of clearance
	const float LEAF_SIZE = 0.1f; // size of voxel when downsamplin using a voxel grid
	const float RADIUS = 2.0;
	const int MIN_NEIGHBORS = 4;

	// compares x and y coordinates of each point to thresh and keeps all points which exceed threshold in either coordinate in either direction
	// threshold is the magnitude of the threshold value
	void box_filter(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output, double threshold = VEHICLE_SIZE)
	{
		ConditionOr<PointType>::Ptr range_cond (new
		  ConditionOr<PointType> ());
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("x", ComparisonOps::GT, threshold)));
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("x", ComparisonOps::LT, -threshold)));
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("y", ComparisonOps::GT, threshold)));
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("y", ComparisonOps::LT, -threshold)));

		ConditionalRemoval<PointType> condrem;
		condrem.setCondition(range_cond);
		condrem.setInputCloud(cloud);
		condrem.filter(*output);
	}

	// same as above, but with ConstPtr input
	void box_filter(PointCloud<PointType>::ConstPtr cloud, PointCloud<PointType>::Ptr output, double threshold = VEHICLE_SIZE)
	{
		ConditionOr<PointType>::Ptr range_cond (new
		  ConditionOr<PointType> ());
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("x", ComparisonOps::GT, threshold)));
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("x", ComparisonOps::LT, -threshold)));
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("y", ComparisonOps::GT, threshold)));
		range_cond->addComparison (FieldComparison<PointType>::ConstPtr (new
		  FieldComparison<PointType> ("y", ComparisonOps::LT, -threshold)));

		ConditionalRemoval<PointType> condrem;
		condrem.setCondition(range_cond);
		condrem.setInputCloud(cloud);
		condrem.filter(*output);
	}

	// remove points above the vihicle
	// compares z coordinate of each point to lower and upper thresholds (which approximate the lowest and highest points of the vehicle) and keep any point in between these thresholds
	void height_filter(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output, float lower_threshold = -HEIGHT_ABOVE_WATER, float upper_threshold = HEIGHT_ABOVE_LIDAR)
	{
		PassThrough<PointType> pass;
		pass.setInputCloud (cloud);
		pass.setFilterFieldName ("z");
		pass.setFilterLimits(lower_threshold, upper_threshold);
		pass.filter(*output);
	}

	// make z coordinates 0
	void projection_filter(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output)
	{
		ModelCoefficients::Ptr coefficients (new ModelCoefficients ());
		coefficients->values.resize (4);
		coefficients->values[0] = coefficients->values[1] = 0;
		coefficients->values[2] = 1.0;
		coefficients->values[3] = 0;

		ProjectInliers<PointType> proj;
		proj.setModelType (SACMODEL_PLANE);
		proj.setInputCloud (cloud);
		proj.setModelCoefficients (coefficients);
		proj.filter (*output);
	}

	// voxel grid
	void downsample(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output, float leaf_size = LEAF_SIZE)
	{
		VoxelGrid<PointType> vg;
		vg.setInputCloud (cloud);
		vg.setLeafSize (leaf_size, leaf_size, leaf_size);
		vg.filter (*output);
	}

	// keep all points which hve a certain number of neighbors whithin a certain distance
	void radius_outlier_filter(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output, float radius = RADIUS, float min_neighbors = MIN_NEIGHBORS)
	{
		RadiusOutlierRemoval<PointType> outrem;
		outrem.setInputCloud(cloud);
		outrem.setRadiusSearch(radius);
		outrem.setMinNeighborsInRadius(min_neighbors);
		outrem.filter (*output);
	}
	
	// extract a subset of a pointcloud
	void extraction_filter(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output, vector<PointIndices>::const_iterator it)
	{
		pcl::ExtractIndices<PointType> extract;
		extract.setInputCloud(cloud);
		extract.setIndices(boost::make_shared<const pcl::PointIndices> (*it));
		extract.filter(*output);
	}

	// filter points outside of a 2d polygon
	void crop_hull(PointCloud<PointType>::Ptr cloud, PointCloud<PointType>::Ptr output, std::vector<pcl::Vertices> vertices, PointCloud<PointType>::Ptr hullcloud = NULL)
	{
		pcl::CropHull<PointType> crhull;
		crhull.setDim(2);
		crhull.setCropOutside(true);
		crhull.setInputCloud(cloud);
		if(hullcloud == NULL)
			crhull.setHullCloud(cloud);
		else
			crhull.setHullCloud(hullcloud);
		crhull.setHullIndices(vertices);
		crhull.filter(*output);
	}
};

class Shape
{
	public:
	pcl::PointCloud<PointType>::Ptr cloud; // point cloud making up the object's 3d shape
	pcl::PointCloud<PointType>::Ptr hull; // point cloud making up the points in the 2d polygon surrounding the object
	std::vector<pcl::Vertices> vertices; // shows ordering of hull points
	
	Shape(){}
	Shape(pcl::PointCloud<PointType>::Ptr c, pcl::PointCloud<PointType>::Ptr h, std::vector<pcl::Vertices> verts)
	{
		cloud = c;
		hull = h;
		vertices = verts;
	}
	
};

const float ALPHA = 4.0f; // controls how tight the concave hull is
const float CLUSTER_TOLERANCE = 2.0f;
const int MIN_CLUSTER_SIZE = 5; // minimum size of a single cluster
const int MAX_CLUSTER_SIZE = 25000; // maximum size of a single cluster

// makes a concave hull around a point cloud
void concave_hull(pcl::PointCloud<PointXYZRGBA>::Ptr cloud, pcl::PointCloud<PointXYZRGBA>::Ptr output, std::vector<pcl::Vertices>* vertices, float alpha = ALPHA)
{
	pcl::ConcaveHull<PointXYZRGBA> chull;
	chull.setKeepInformation(true); 
	chull.setDimension(2);
	chull.setInputCloud (cloud);
	chull.setAlpha (alpha);
	chull.reconstruct (*output, *vertices);
}

// uses euclidean clustering to split a point cloud into clusters
// indices of the clusters are stored in cluster_indices
void cluster(pcl::PointCloud<PointXYZRGBA>::Ptr cloud, std::vector<PointIndices>* cluster_indices, float cluster_tolerance = CLUSTER_TOLERANCE, int min_cluster_size = MIN_CLUSTER_SIZE, int max_cluster_size = MAX_CLUSTER_SIZE)
{
	search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new search::KdTree<pcl::PointXYZRGBA>);
        tree->setInputCloud (cloud);

        EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
        ec.setClusterTolerance (cluster_tolerance);
        ec.setMinClusterSize (min_cluster_size);
        ec.setMaxClusterSize (max_cluster_size);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud);
        ec.extract (*cluster_indices);
}

void get_shapes_from_cloud(vector<Shape*>* shapes, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr unfiltered_ipt = NULL)
{
	// set the unfiltered cloud and copy it
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr unfiltered (new pcl::PointCloud<PointType>);
	if (unfiltered_ipt == NULL)
		copyPointCloud(*cloud, *unfiltered);
	else
		copyPointCloud(*unfiltered_ipt, *unfiltered);

	// get clusters and store their indices in cluster_indices
        vector<PointIndices> cluster_indices;
        cluster(cloud, &cluster_indices);

	// loop over clusters	
	for (vector<PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) 
	{
		// extract the indices of the cluster, making a point cloud of just the cluster
		pcl::PointCloud<PointType>::Ptr hull (new pcl::PointCloud<PointType>);
		Filters::extraction_filter(cloud, hull, it);
		
		// create a concave hull around the cluster and store the boundary in a new point cloud/ vector of vertices
		std::vector<pcl::Vertices> verts;
		concave_hull(hull, hull, &verts);
		
		// offset the vertices so that they're correct after concatenating the hull to the unfiltered cloud
		std::vector<pcl::Vertices> verts_offset (verts);
		for(int i = 0; i < verts_offset.size(); i++)
		{
			for(int j = 0; j < verts_offset[i].vertices.size(); j++)
			{
				verts_offset[i].vertices[j] += unfiltered->size();
			}
		}
		
		// concatenate the hull to the unfiltered point cloud, and get points lying inside the hull
		*unfiltered += *hull;
		pcl::PointCloud<PointType>::Ptr crop_cloud (new pcl::PointCloud<PointType>);
		Filters::crop_hull(unfiltered, crop_cloud, verts_offset);
		
		// make a new Shape and add it to the shapes vector
		Shape* shape = new Shape(crop_cloud, hull, verts);
		shapes->push_back(shape);
	}
}

void visualize_shape(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, Shape* shape, bool show_hull = true, bool show_cloud = true, bool keep_color = true)
{
	// create address to append to the id string
	char buffer [50];
	sprintf(buffer, "%p", shape);
	std::string address = buffer;
	
	// if showing the hull, make a PolygonMesh and add the visualization to viewer
	if (show_hull)
	{
		pcl::PolygonMesh mesh;
		mesh.polygons = shape->vertices;
		pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2);
		pcl::toPCLPointCloud2(*(shape->hull), *cloud_blob);
		mesh.cloud = *cloud_blob;
		viewer->addPolylineFromPolygonMesh(mesh, "polygon_" + address);
	}
	
	// if showing the cloud:
	if (show_cloud)
	{
		// randomly pick a distinct color that's not black
		uint8_t r = 255 * (rand() % 2);
		uint8_t g = 255 * (rand() % 2);
		uint8_t b = r == 0 && g == 0 ? 255 : 255 * (rand() % 2);
		uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
	     	static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
		
		// set the color and add the point cloud to viewer
		visualization::PointCloudColorHandlerCustom<PointXYZRGBA> handler(shape->cloud, r, g, b);
		viewer->addPointCloud<pcl::PointXYZRGBA> (shape->cloud, handler, "shape_" + address);
  		viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "shape_" + address);
	}
}


int main(int argc, char *argv[]) 
{
	// Command-Line Argument Parsing
	if( console::find_switch( argc, argv, "-help" ))
	{
		cout << "usage: " << argv[0]
		//            << " [-calibrationFile]"
		    << " [-pcap <*.pcap>]"
		    << " [-help]"
		    << endl;
		return 0;
	}
    string pcap;
    // parse_argument (argc, argv, "-calibrationFile", hdlCalibration);
    console::parse_argument( argc, argv, "-pcap", pcap );

    // Point Cloud
    PointCloud<PointType>::ConstPtr cloud (new PointCloud<PointType>);
	std::vector<Shape*> shapes;

    // PCL Visualizer
    boost::shared_ptr<visualization::PCLVisualizer> viewer( new visualization::PCLVisualizer( "Velodyne Viewer" ) );
    viewer->addCoordinateSystem( 3.0, "coordinate" );
    viewer->setBackgroundColor( 0.0, 0.0, 0.0, 0 );
    viewer->initCameraParameters();
    viewer->setCameraPosition( 0.0, 0.0, 30.0, 0.0, 1.0, 0.0, 0 );

    // Retrieved Point Cloud Callback Function
    boost::mutex mutex;
    boost::function<void( const PointCloud<PointType>::ConstPtr& )> function;
    function = [ &cloud, &shapes, &mutex ](const PointCloud<PointType>::ConstPtr& ptr )
	{
		boost::mutex::scoped_lock lock( mutex ); // for multi-threading

		/* Point Cloud Processing */

		// delete old shapes and clear the vector
		for(int i = 0; i < shapes.size(); i++)
			delete shapes[i];
		shapes.clear();

		PointCloud<PointType>::Ptr cloud_postfilter (new PointCloud<PointType>);
		cloud = ptr;

		// Filter out all points within threshold box from vehicle (origin)
		Filters::box_filter(cloud, cloud_postfilter);

		// Filter out points that are higher than vehicle
		Filters::height_filter(cloud_postfilter, cloud_postfilter);

		// Project all points down to XY plane
		Filters::projection_filter(cloud_postfilter, cloud_postfilter);

		// Downsample the dataset using voxel grids
		Filters::downsample(cloud_postfilter, cloud_postfilter);

		// Filter out all points which don't have many neighbors within a particular radius
		Filters::radius_outlier_filter(cloud_postfilter, cloud_postfilter);
		
		// get shapes from the cloud
		get_shapes_from_cloud(&shapes, cloud_postfilter, ptr);

	};

    // VLP Grabber
    boost::shared_ptr<HDLGrabber> grabber;
    if( !pcap.empty() ) 
	{
        cout << "Capture from PCAP..." << endl;
        grabber = boost::shared_ptr<HDLGrabber>( new HDLGrabber( "", pcap ) );
    }

    // Register Callback Function
    boost::signals2::connection connection = grabber->registerCallback( function );

    // Start Grabber
    grabber->start();

	while( !viewer->wasStopped())
	{
        // Update Viewer
        viewer->spinOnce();

        boost::mutex::scoped_try_lock lock( mutex );

        if(lock.owns_lock()) 
		{
			// clear the visualiation
			viewer->removeAllShapes();
			viewer->removeAllPointClouds();

			// visualize the new shapes
			for(int i = 0; i < shapes.size(); i++)
			{
				visualize_shape(viewer, shapes[i]);
			}

			// visualize the original point cloud with smaller white points
			visualization::PointCloudColorHandlerCustom<PointXYZRGBA> rgb(cloud, 255, 255, 255);
			viewer->addPointCloud(cloud, rgb, "cloud");
			viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "cloud");
        }
    }

    // Stop Grabber
    grabber->stop();

    // Disconnect Callback Function
    if( connection.connected()){
        connection.disconnect();
    }

    return 0;
}
