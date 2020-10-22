/* 
Name: Rasika Hedaoo
Student ID: 1001770527
*/

import java.io.*;
import java.util.Vector;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

class Vertex implements Writable {
	public long id; 					// the vertex ID
	public Vector<Long> adjacent;		// the vertex neighbors
	public long centroid;				// the id of the centroid in which this vertex belongs to
	public long size;
	public short depth; 				// the BFS depth

	
  Vertex(long id, Vector<Long> adjacent, long centroid, short depth) {
    this.id = id;
    this.adjacent = adjacent;
    this.centroid = centroid;
    this.depth = depth;
    size = adjacent.size();
  }

  public Vertex() {}

  public void readFields(DataInput inpt) throws IOException {
    id = inpt.readLong();
    adjacent = new Vector<Long>();
    centroid = inpt.readLong();
    depth = inpt.readShort();
    size = inpt.readLong();
    for (int y = 0; y < size; y++) {
		adjacent.add(inpt.readLong());
    }
  }

  public void write(DataOutput outpt) throws IOException {
    outpt.writeLong(id);
    outpt.writeLong(centroid);
    outpt.writeShort(depth);
    outpt.writeLong(adjacent.size());
    for (int i = 0; i < adjacent.size(); i++) {
      outpt.writeLong(adjacent.get(i));
    }
  }
}

public class GraphPartition {
	static Vector<Long> otherCentroids = new Vector<Long>();
	final static short max_depth = 8;
	static short BFS_depth = 0;
/* The first Map-Reduce job is to read the graph 

map ( key, line ) =
  parse the line to get the vertex id and the adjacent vector
  // take the first 10 vertices of each split to be the centroids
  for the first 10 vertices, centroid = id; for all the others, centroid = -1
  emit( id, new Vertex(id,adjacent,centroid,0) )
*/
  public static class GraphMapper extends Mapper<Object, Text, LongWritable, Vertex> {
    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
		long centroid;
		String line = value.toString();
		String[] vertices = line.split(",");

		long vertexId = Long.parseLong(vertices[0]);

		Vector<Long> adjacent = new Vector<Long>();
		for (int i = 1; i < vertices.length; i++) {
			Long adj = Long.parseLong(vertices[i]);
			adjacent.add(adj);
		}
		if (otherCentroids.size() < 10) {
			otherCentroids.add(vertexId);
			centroid = vertexId;
		} else {
			otherCentroids.add(vertexId);
			centroid = -1;
      }
      context.write(new LongWritable(vertexId), new Vertex(vertexId, adjacent, centroid, (short) 0));
    }
  }

/*The second Mapper job is to do BFS:

map ( key, vertex ) =
  emit( vertex.id, vertex )   // pass the graph topology
  if (vertex.centroid > 0)
     for n in vertex.adjacent:     // send the centroid to the adjacent vertices
        emit( n, new Vertex(n,[],vertex.centroid,BFS_depth) ) 
*/
  public static class BFSMapper extends Mapper<LongWritable, Vertex, LongWritable, Vertex> {
    @Override
    public void map(LongWritable key, Vertex vertex, Context context) 
			throws IOException, InterruptedException {
		context.write(new LongWritable(vertex.id), vertex);
		if (vertex.centroid > 0) {
			for (Long n : vertex.adjacent) {
				context.write(new LongWritable(n), new Vertex(n, new Vector<Long>(), vertex.centroid, BFS_depth));
			}
		}
    }

  }
/*The Reducer job is to do BFS:

reduce ( id, values ) =
  min_depth = 1000
  m = new Vertex(id,[],-1,0)
  for v in values:
     if (v.adjacent is not empty)
        m.adjacent = v.adjacent
     if (v.centroid > 0 && v.depth < min_depth)
        min_depth = v.depth
        m.centroid = v.centroid
  m.depth = min_depth
  emit( id, m )
*/
  public static class BFSReducer extends Reducer<LongWritable, Vertex, LongWritable, Vertex> {
    @Override
    public void reduce(LongWritable vertexId, Iterable<Vertex> values, Context context)
        throws IOException, InterruptedException {
		short min_depth = 1000;
		Vertex m = new Vertex(vertexId.get(), new Vector<Long>(), (long) (-1), (short) (0));
		for (Vertex v : values) {
			if (!(v.adjacent.isEmpty())) {
				m.adjacent = v.adjacent;
			}
			if (v.centroid > 0 && v.depth < min_depth) {
				min_depth = v.depth;
				m.centroid = v.centroid;
			}
		}
		m.depth = min_depth;
		context.write(vertexId, m);
    }
  }
/* The final Map-Reduce job is to calculate the cluster sizes:

map ( id, value ) =
   emit(value.centroid,1)
*/
  public static class ClusterMapper extends Mapper<LongWritable, Vertex, LongWritable, IntWritable> {
    @Override
    public void map(LongWritable vertexId, Vertex value, Context con) throws IOException, InterruptedException {
		con.write(new LongWritable(value.centroid), new IntWritable(1));
    }

  }

/* The Reduce job is to calculate the cluster sizes:

reduce ( centroid, values ) =
   m = 0
   for v in values:
       m = m+v
   emit(centroid,m)
*/
  public static class ClusterReducer extends Reducer<LongWritable, IntWritable, LongWritable, LongWritable> {
    @Override
    public void reduce(LongWritable centroid, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
		long m = 0L;
		for (IntWritable v : values) {
			m = m + Long.valueOf(v.get());
		}
		context.write(centroid, new LongWritable(m));
    }
  }

  /* ... */

  public static void main(String[] args) throws Exception {
    Job job = Job.getInstance();
	/* First Map-Reduce job for reading the graph */
    job.setJobName("GraphData");
    job.setJarByClass(GraphPartition.class);
    job.setMapperClass(GraphMapper.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(Vertex.class);
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(Vertex.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(job,  new Path(args[0]));			//args[0] is the input graph
	FileOutputFormat.setOutputPath(job, new Path(args[1]+"/i0"));	//args[1] is the intermediate directory. first Map-Reduce job writes on the directory args[1]+"/i0"
    job.waitForCompletion(true);									

    Path inpath = new Path(args[1]);
    for (short i = 0; i < max_depth; i++) {	//i is the for-loop index you use to repeat the second Map-Reduce job
      BFS_depth++;							//The variable BFS_depth is bound to the iteration number (from 1 to 8)
      job = Job.getInstance();
	  /* Second Map-Reduce job for BFS */
      job.setJobName("MappingMethod");
      job.setJarByClass(GraphPartition.class);
      job.setOutputKeyClass(LongWritable.class);
      job.setOutputValueClass(Vertex.class);
      job.setMapOutputKeyClass(LongWritable.class);
      job.setMapOutputValueClass(Vertex.class);
      job.setMapperClass(BFSMapper.class);
      job.setReducerClass(BFSReducer.class);
      job.setInputFormatClass(SequenceFileInputFormat.class);
	  job.setOutputFormatClass(SequenceFileOutputFormat.class);
	  FileInputFormat.setInputPaths(job, new Path(args[1]+"/i"+i));				//Map-Reduce job reads from the directory args[1]+"/i"+i ..
      FileOutputFormat.setOutputPath(job, new Path(args[1] + "/i" + (i + 1)));	//and writes in the directory args[1]+"/i"+(i+1)
      job.waitForCompletion(true);
    }

    job = Job.getInstance();
    job.setJobName("GraphLength");
	/* Third Map-Reduce job to calculate the cluster sizes */
    job.setJarByClass(GraphPartition.class);
    job.setMapperClass(ClusterMapper.class);
    job.setReducerClass(ClusterReducer.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(LongWritable.class);
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(IntWritable.class);
	job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    FileInputFormat.setInputPaths(job, new Path(args[1]+"/i8"));	//The final Map-Reduce job reads from args[1]+"/i8" and writes on args[2]
	FileOutputFormat.setOutputPath(job, new Path(args[2]));			//args[2] is the output
    job.waitForCompletion(true);
  }
}