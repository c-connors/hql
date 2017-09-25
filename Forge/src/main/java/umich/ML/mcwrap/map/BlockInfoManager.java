package umich.ML.mcwrap.map;

import net.minecraft.entity.Entity;
import net.minecraft.entity.passive.*;
import net.minecraft.entity.monster.*;

import net.minecraft.block.Block;
import net.minecraft.util.BlockPos;
import net.minecraft.block.properties.IProperty;
import net.minecraft.block.state.IBlockState;
import net.minecraft.item.EnumDyeColor;
import net.minecraftforge.fml.common.FMLCommonHandler;

import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.NodeList;
import umich.ML.mcwrap.util.FileParser;
import umich.ML.mcwrap.MCWrap;
import umich.ML.mcwrap.map.BlockEntity;

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**************************************************
 * Package: umich.ML.mcwrap.map
 * Class: BlockInfoManager
 * Timestamp: 5:36 PM 12/19/15
 * Authors: Valliappa Chockalingam, Junhyuk Oh
 **************************************************/

public class BlockInfoManager {

    private static final HashMap<Integer, BlockEntity> idToBlock = new HashMap<Integer, BlockEntity>();
    private static final HashMap<Integer, List<BlockEntity>> idToEntity = new HashMap<Integer, List<BlockEntity>>();
    private static final HashMap<Integer, String> idToName = new HashMap<Integer, String>();

    static IBlockState defaultBlockState = null;
    static int defaultBlockId = 0;

    public static void init(String filePath)
    {
    	idToBlock.clear();
    	//idToEntity.clear();
    	idToName.clear();
    	
        Element objectXML = FileParser.readXML(filePath);

        NodeList nList = objectXML.getElementsByTagName("object");

        for(int i = 0; i < nList.getLength(); i++)
        {
            NamedNodeMap namedNodeMap = nList.item(i).getAttributes();
            
            int objectId = Integer.parseInt(namedNodeMap.getNamedItem("id").getNodeValue());
            String objectType = namedNodeMap.getNamedItem("type").getNodeValue();
            
            if (objectType.equals("block")) {
            	IBlockState blockState = null;
            	try {
            		blockState = Block.getBlockFromName(namedNodeMap.getNamedItem("obj_type").getNodeValue()).getDefaultState();
            	} catch (Exception e) {
            		System.out.println("Invalid block type: " + namedNodeMap.getNamedItem("obj_type").getNodeValue());
            	}
	            if(namedNodeMap.getNamedItem("color") != null)
	            {
	                String color = namedNodeMap.getNamedItem("color").getNodeValue();
	
	                List properties = blockState.getProperties().keySet().asList();
	
	                //noinspection ForLoopReplaceableByForEach
	                for (int j = 0; j < properties.size(); j++) {
	                    IProperty prop = (IProperty) properties.get(j);
	                    if (prop.getName().equals("color") && prop.getValueClass() == net.minecraft.item.EnumDyeColor.class)
	                        blockState = blockState.withProperty(prop, EnumDyeColor.valueOf(color));
	                }
	            }
	
	            if(namedNodeMap.getNamedItem("brightness") != null)
	            {
	                Float brightness = Float.parseFloat(namedNodeMap.getNamedItem("brightness").getNodeValue());
	
	                blockState.getBlock().setLightLevel(brightness);
	            }
	
	            if(namedNodeMap.getNamedItem("default") != null) {
	                defaultBlockState = blockState;
	                defaultBlockId = objectId;
	            }
	
	            idToBlock.put(objectId, new BlockEntity(objectId, blockState));
            }
            else if (objectType.equals("mob")) {
            	String mobName = namedNodeMap.getNamedItem("obj_type").getNodeValue();
            	idToName.put(objectId, mobName);
            }
            else {
            	System.out.print("Invalid object type: " + objectType);
                FMLCommonHandler.instance().exitJava(-1, false);
            }
        }
    }
    
    public static BlockEntity getBlockEntity(BlockPos pos, int id) {
    	BlockEntity ent = getEntity(id);
    	//ent.addEntityToWorld(pos);
    	return ent;
    }
    
    public static IBlockState getBlockStatebyId(int id) {
    	return idToBlock.get(id).block;
    }
    
    public static BlockEntity createEntity(int id) {
    	BlockEntity blockent = idToBlock.get(id);
    	if (blockent != null) {
    		return blockent;
    	}
    	
    	String object_name = idToName.get(id);
    	
    	if (object_name == null) {
    		System.out.println("Invalid object id: " + id);
    	}
    	if (object_name.equals("chicken")) {
    		EntityChicken ent = new EntityChicken(MCWrap.world);
    		ent.setGrowingAge(4);
    		return new BlockEntity(id, ent);
    	} else if(object_name.equals("cow")) {
    		EntityCow ent = new EntityCow(MCWrap.world);
    		return new BlockEntity(id, ent);
    	} else if(object_name.equals("wolf")) {
    		EntityWolf ent = new EntityWolf(MCWrap.world);
    		ent.setCollarColor(EnumDyeColor.valueOf("RED"));
    		return new BlockEntity(id, ent);
    	} else if(object_name.equals("cat")) {
    		EntityOcelot ent = new EntityOcelot(MCWrap.world);
    		ent.setTameSkin(0);
    		//ent.setGrowingAge(-1);
    		return new BlockEntity(id, ent);
    	} else if(object_name.equals("pig")) {
    		EntityPig ent = new EntityPig(MCWrap.world);
    		return new BlockEntity(id, ent);
    	} else if(object_name.equals("sheep")) {
    		EntitySheep ent = new EntitySheep(MCWrap.world);
    		//ent.setFleeceColor(EnumDyeColor.valueOf("BLACK"));
    		return new BlockEntity(id, ent);
		} else if(object_name.equals("horse")) {
    		EntityHorse ent = new EntityHorse(MCWrap.world);
    		ent.setHorseType(2);
    		return new BlockEntity(id, ent);
		} else if(object_name.equals("rabbit")) {
			EntityRabbit ent = new EntityRabbit(MCWrap.world);
    		return new BlockEntity(id, ent);
		} else if(object_name.equals("zombie")) {
			EntityZombie ent = new EntityZombie(MCWrap.world);
    		return new BlockEntity(id, ent);
		} else if(object_name.equals("creeper")) {
			EntityCreeper ent = new EntityCreeper(MCWrap.world);
    		return new BlockEntity(id, ent);
		} else if(object_name.equals("spider")) {
			EntitySpider ent = new EntitySpider(MCWrap.world);
    		return new BlockEntity(id, ent);
		} else {
			System.out.println("Invalid object name: " + object_name);
			return null;
    	}
    }
    
    public static void freeEntity(BlockPos pos, BlockEntity ent) {
    	ent.removeFromWorld(pos);
    	if (ent.isBlock()) {
	    	List<BlockEntity> entity_list = idToEntity.get(ent.id);
	    	entity_list.add(0, ent);
    	}
    }
    
    public static BlockEntity getEntity(int id) {
    	if (idToBlock.containsKey(id)) {
	    	List<BlockEntity> entity_list = idToEntity.get(id);
	    	if (entity_list == null) {
	    		entity_list = new ArrayList<BlockEntity>();
	    		idToEntity.put(id, entity_list);
	    	}
	    	if (entity_list.isEmpty()) {
	    		entity_list.add(0, createEntity(id));
	    	} 
	    	return entity_list.remove(0);
    	} else {
    		return createEntity(id);
    	}
    }
    
    public static void cleanAllEntities() {
    	idToEntity.clear();
    }
    
    
    public static int size() {
    	int size = 0;
    	for (Integer id : idToEntity.keySet()) {
    		size = size + idToEntity.get(id).size();
    	}
    	return size;
    }
}
