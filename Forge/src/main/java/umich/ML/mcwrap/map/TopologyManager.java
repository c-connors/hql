package umich.ML.mcwrap.map;

import net.minecraft.entity.monster.EntitySkeleton;
import net.minecraft.entity.monster.EntityCreeper;
import net.minecraft.entity.passive.EntityPig;
import net.minecraft.entity.item.EntityItem;
import net.minecraft.entity.projectile.EntityEgg;
import net.minecraft.block.state.IBlockState;
import net.minecraft.util.BlockPos;
import net.minecraftforge.fml.common.FMLCommonHandler;
import umich.ML.mcwrap.MCWrap;
import umich.ML.mcwrap.configuration.ConfigurationHandler;
import umich.ML.mcwrap.task.Task;
import umich.ML.mcwrap.util.FileParser;

import java.io.File;
import java.util.*;

/**************************************************
 * Package: umich.ML.mcwrap.map
 * Class: TopologyManager
 * Timestamp: 5:17 PM 12/19/15
 * Authors: Valliappa Chockalingam, Junhyuk Oh
 **************************************************/

@SuppressWarnings("unused")
public class TopologyManager {

	private static HashMap<BlockPos, BlockEntity> blockEntities = new HashMap<BlockPos, BlockEntity>();
	
	public static void reset() 
	{
		if (MCWrap.world != null) 
			MCWrap.world.markBlockRangeForRenderUpdate(new BlockPos(0, 0, 0), new BlockPos(10, 10, 10));
	}
	
    private static ArrayList<BlockPos> parseTopology(String data)
    {
    	int mapSize = ConfigurationHandler.getMapSize();
    	String[] arr = data.split(",");
        ArrayList<BlockPos> blocksInTopology = new ArrayList<BlockPos>();
        int maxY = 0;
        try
        {
            for (int i = 0; i < mapSize; i++)
                for (int j = 0; j < mapSize; j++)
                    for (int k = 0; k < Integer.parseInt(arr[i*mapSize + j]); k++) {
                        blocksInTopology.add(new BlockPos(j, k, i));
                        if(Integer.parseInt(arr[i*mapSize + j]) > maxY)
                            maxY = Integer.parseInt(arr[i*mapSize + j]);
                    }

            if(ConfigurationHandler.getRoofEnable())
                for (int i = 0; i < mapSize; i++)
                    for (int j = 0; j < mapSize; j++) {
                        blocksInTopology.add(new BlockPos(j, maxY, i));
                    }
        }

        catch(NumberFormatException e)
        {
            System.out.print("Number format exception on parsing Topology!");
            System.out.print(data);
            FMLCommonHandler.instance().exitJava(-1, false);
        }

        return blocksInTopology;
    }
    
    private static HashMap<BlockPos, Integer> parseObjects(String data)
    {
    	int mapSize = ConfigurationHandler.getMapSize();
    	String[] arr = data.split(",");
    	HashMap<BlockPos, Integer> objects = new HashMap<BlockPos, Integer>();
        try
        {
            for (int i = 0; i < mapSize; i++)
                for (int j = 0; j < mapSize; j++) {
                	if (!arr[i*mapSize +j].equals(" ") && !arr[i*mapSize +j].equals("0")) {
                		int object_id = Integer.parseInt(arr[i*mapSize + j]);
	                	objects.put(new BlockPos(j, ConfigurationHandler.getObjectY(), i), object_id);
                	}
                }
        }
        catch(NumberFormatException e)
        {
            System.out.print("Number format exception on parsing Objects!");
            System.out.print(data);
            FMLCommonHandler.instance().exitJava(-1, false);
        }

        return objects;
    }
    
    public static void update() {
    	//System.out.println("Block entities size: " + blockEntities.size() + " Cache size: " + BlockInfoManager.size());
    	for(BlockPos pos : blockEntities.keySet()) {
    		blockEntities.get(pos).update();
    	}
    }
    
    public static void updateTopology(String topology_data, String objects_data)
    {   
        ArrayList<BlockPos> new_topology = parseTopology(topology_data);
        HashMap<BlockPos, Integer> new_object_id = parseObjects(objects_data);

        for(BlockPos pos : new_object_id.keySet())
            new_topology.remove(pos);
        
    	if (MCWrap.world == null)
    		System.out.println("MCWrap.world is null!");
    	
    	
    	Set<BlockPos> pos_set = new TreeSet<BlockPos>(blockEntities.keySet());
    
        for(BlockPos pos : pos_set)
        {
        	Integer obj_id = new_object_id.get(pos);
        	
            if(!new_topology.contains(pos) && obj_id == null) {
            	BlockEntity ent = blockEntities.remove(pos);
            	BlockInfoManager.freeEntity(pos, ent);
            }

            else if(!new_topology.contains(pos) && obj_id != null) {
            	BlockEntity ent = blockEntities.get(pos);
            	if (ent.id != obj_id) {
                	blockEntities.remove(ent);
                	BlockInfoManager.freeEntity(pos, ent);
                	
                	BlockEntity new_ent = BlockInfoManager.getBlockEntity(pos, obj_id);
                	blockEntities.put(pos, new_ent);
                	new_ent.addEntityToWorld(pos);
            	}
            }
            else if(new_topology.contains(pos)  && obj_id == null) {
            	BlockEntity ent = blockEntities.get(pos);
            	if (ent.id != BlockInfoManager.defaultBlockId) {
            		//System.out.println(ent.id + " " + BlockInfoManager.defaultBlockId);
            		//System.out.format("%d, %d, %d\n", pos.getX(), pos.getY(), pos.getZ());
            		
                	blockEntities.remove(ent);
                	BlockInfoManager.freeEntity(pos, ent);
                	
                	BlockEntity new_ent = BlockInfoManager.getBlockEntity(pos, 
                				BlockInfoManager.defaultBlockId);
                	blockEntities.put(pos, new_ent);
                	new_ent.addEntityToWorld(pos);
            	}
            }
            else FMLCommonHandler.instance().exitJava(-1, false);
            MCWrap.world.markBlockForUpdate(pos);
        }
    	
        for(BlockPos pos : new_topology)
        {
        	BlockEntity ent = blockEntities.get(pos);
            if(ent == null) {
                ent = BlockInfoManager.getBlockEntity(pos, 
                			BlockInfoManager.defaultBlockId);
                blockEntities.put(pos, ent);
                //MCWrap.world.setBlockState(pos, BlockInfoManager.defaultBlockState);
                MCWrap.world.markBlockForUpdate(pos);
            }
            MCWrap.world.setBlockState(pos, BlockInfoManager.defaultBlockState);
        }

        for(BlockPos pos : new_object_id.keySet())
        {
        	BlockEntity ent = blockEntities.get(pos);
            if(ent == null) {
            	ent = BlockInfoManager.getBlockEntity(pos, new_object_id.get(pos));
                blockEntities.put(pos, ent);
                ent.addEntityToWorld(pos);
                MCWrap.world.markBlockForUpdate(pos);
            }
        }
    }
}
