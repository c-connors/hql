package umich.ML.mcwrap.communication;

import net.minecraftforge.fml.common.FMLCommonHandler;
import umich.ML.mcwrap.configuration.ConfigurationHandler;
import umich.ML.mcwrap.task.ActionHandler;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;

/**************************************************
 * Package: umich.ML.mcwrap.communication
 * Class: CommunicationHandler
 * Timestamp: 2:51 PM 12/20/15
 * Authors: Valliappa Chockalingam, Junhyuk Oh
 **************************************************/

public class CommunicationHandler {
    private static Boolean initialized = false;

    private static ServerSocket server;
    private static Socket client;
    private static OutputStream dataSender;
    private static BufferedReader dataReceiver;
    
    public static ServerSocket getServerSocket(int given_port)
    {
    	if (given_port > 0) {
	    	try
			{
				ServerSocket s = new ServerSocket(given_port);
				return s;      // Created okay, so this port is available.
			}
			catch (Exception e)
			{
				return null;
			}
    	} else {
    		ServerSocket s = null;
        	int port = 30000;
        	while (s == null && port <= 31000)
        	{
        		try
        		{
        			s = new ServerSocket(port);
        			return s;      // Created okay, so this port is available.
        		}
        		catch (Exception e)
        		{
        			// Try the next port.
              port++;
        		}
        	}
        	return null;   // No port found in the allowed range.
    	}
    }

    public static void init()
    {
        if(!initialized) {
            System.out.print("[[ Communication Handler  (( init )) ]] : No singleton, " +
                    "creating an instance for the first time! Proceeding.\n");

            try {
                server = getServerSocket(ConfigurationHandler.getPort());

                server.setSoTimeout(0);

                System.out.print("[[ Communication Handler  (( init )) ]] : " +
                        "Listening for connections on port " + server.getLocalPort());

                client = server.accept();

                System.out.print("[[ Communication Handler  (( init )) ]] : Connected to " +
                		client.getRemoteSocketAddress().toString() + ".\n");

                dataSender = client.getOutputStream();
                dataReceiver = new BufferedReader(new InputStreamReader(client.getInputStream()));

            } catch (Exception e) {
                System.out.print("[[ Communication Handler  (( init )) ]] : " +
                        "Caught exception on creation of socket communication handler!\n");
                FMLCommonHandler.instance().exitJava(-1, false);
            }

            initialized = true;
        }

        else System.out.print("[[ Communication Handler (( init )) ]] : Singleton exists, " +
                "skipping init.\n");
    }
    
    public static void reconnect() {
    	try {
    		client.close();
            server.close();
    		server = getServerSocket(ConfigurationHandler.getPort());

            server.setSoTimeout(0);

            System.out.print("[[ Communication Handler  (( init )) ]] : " +
                    "Listening for connections on port " + server.getLocalPort());

            client = server.accept();

            System.out.print("[[ Communication Handler  (( init )) ]] : Connected to " +
            		client.getRemoteSocketAddress().toString() + ".\n");

            dataSender = client.getOutputStream();
            dataReceiver = new BufferedReader(new InputStreamReader(client.getInputStream()));

        } catch (Exception e) {
            System.out.print("[[ Communication Handler  (( init )) ]] : " +
                    "Caught exception on creation of socket communication handler!\n");
            FMLCommonHandler.instance().exitJava(-1, false);
        }
    }

    public static void send(byte[] message)
    {
        try
        {
            dataSender.write(message);
        }

        catch(Exception e)
        {
            System.out.println("[[ Communication Handler  (( send )) ]] : Caught exception on sending message!\n");
            FMLCommonHandler.instance().exitJava(-1, false);
        }
    }

    public static void send(String message)
    {
        send(message.getBytes());
    }

    public static String receive() throws Exception
    {
        String message = null;

        try
        {
            message = dataReceiver.readLine();
        }

        catch(Exception e)
        {
            System.out.println("[[ Communication Handler  (( init )) ]] : Caught exception on receiving message!");
            System.out.println(e.getMessage());
            throw e;
            //FMLCommonHandler.instance().exitJava(-1, false);
        }

        return message;
    }
    
    public static String getTask()
    {
    	String task = null;
    	
    	try { 
    		task = receive();
    	}
    	catch(Exception e) {
    		 System.out.println("[[ Communication Handler  (( init )) ]] : Caught exception on receiving message!");
             System.out.println(e.getMessage());
             FMLCommonHandler.instance().exitJava(-1, false);
    	}
    	assert(task.startsWith("set_task"));

        if(task.contains(" ")){
            System.out.println("[[ Communication Handler  (( getTask )) ]] : " +
                    "Task string contains a space, assuming that a Boolean follows which specifies " +
                    "whether possible actions are to be sent.\n");
            String arr[] = task.split(" ");
            System.out.println("[[ Communication Handler  (( getTask )) ]] : " +
                    "Recorded that possible actions to be sent?: " + arr[1] + " Notifying the ActionHandler.\n");
            ActionHandler.shouldSendPossibleActions(Boolean.parseBoolean(arr[1]));
            System.out.println("[[ Communication Handler  (( getTask )) ]] : " +
                    "ActionHandler notified.\n");
        }
        else
        {
            System.out.println("[[ Communication Handler  (( getTask )) ]] : " +
                    "Task string contains no spaces, assuming that possible action are NOT to be sent. " +
                    "Notifying the ActionHandler.\n");
            ActionHandler.shouldSendPossibleActions(false);
            System.out.println("[[ Communication Handler  (( getTask )) ]] : " +
                    "ActionHandler notified.\n");
        }

        return task.substring("set_task_".length());
    }

    public static void handshake(int displayHeight, int displayWidth,
                          int numChannels)
    {
        send((Integer.toString(numChannels) + " " +
                Integer.toString(displayHeight) + " " +
                Integer.toString(displayWidth) + "\n"));
    }
}
