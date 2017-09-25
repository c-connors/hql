package umich.ML.mcwrap.task;


import umich.ML.mcwrap.configuration.ConfigurationHandler;
import umich.ML.mcwrap.goal.GoalFSM;
import umich.ML.mcwrap.logger.Logger;
import umich.ML.mcwrap.map.*;
import net.minecraft.util.BlockPos;

import java.util.Random;

/**************************************************
 * Package: umich.ML.mcwrap.task
 * Class: Task
 * Timestamp: 4:16 PM 12/19/15
 * Authors: Valliappa Chockalingam, Junhyuk Oh
 **************************************************/

public class Task {
    private static Boolean initialized = false;

    private Task() {}

    public static void init(String RootTaskDir) {
        //if(!initialized) {
            System.out.print("[[ Task (( init )) ]] : No singleton, " +
                    "creating an instance for the first time! Proceeding.\n");

            System.out.print("[[ Task (( init )) ]] : Checking for task file: " +
                    RootTaskDir + "/task.xml" + ".\n");
            ConfigurationHandler.initTask(RootTaskDir + "/task.xml");
            System.out.print("[[ Task (( init )) ]] : Finished logging task file.\n");

            System.out.print("[[ Task (( init )) ]] : Initializing Action Handler with file: " +
                    RootTaskDir + "/actions.xml" + ".\n");
            ActionHandler.init(RootTaskDir + "/actions.xml");
            System.out.print("[[ Task (( init )) ]] : Finished Action Handler initialization.\n");

            System.out.print("[[ Task (( init )) ]] : Initializing Block Info. Manager with file: " +
                    RootTaskDir + "/blockTypeInfo.xml" + ".\n");
            BlockInfoManager.init(RootTaskDir + "/object.xml");
            System.out.print("[[ Task (( init )) ]] : Finished Block Info. Manager initialization.\n");
            
            TopologyManager.reset();
            initialized = true;
        //}
        //else System.out.print("[[ Task (( init )) ]] : Singleton exists, " +
        //        "skipping init.\n");
    }

    public static void reset(String topology, String objects, String player)
    {
    	//System.out.println("Topology: " + topology);
    	//System.out.println("Object: " + objects);
    	//System.out.println("Player: " + player);
    	
    	TopologyManager.updateTopology(topology, objects);
    	String[] pos_info = player.split(",");
    	int pos_z = Integer.parseInt(pos_info[0]);
    	int pos_x = Integer.parseInt(pos_info[1]);
    	int yaw = Integer.parseInt(pos_info[2]);
    	int pitch = Integer.parseInt(pos_info[3]);
    	
    	PlayerState.logPosChange(new BlockPos(pos_x, ConfigurationHandler.getPlayerY(), pos_z));
    	PlayerState.logFlying(false);
      	PlayerState.logPitchChange(-pitch * 45.0F);
      	PlayerState.logYawChange(yaw * 90.0F);
      	
      	PlayerState.syncPlayer();
      	TopologyManager.update();
    }
}
