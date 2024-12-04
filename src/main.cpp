#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>

// glm
#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

// stb
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <set>
#include <fstream>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 800; // 600
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    // has to be able to interface with a swapchain to display (ex no non-display gpus)
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    // fixes VUID-VkDeviceCreateInfo-pProperties-04451, also makes use of the beta header cuz im cool like that
    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

// structs

// alignas(16) forces the attribute to take up 16 bytes of memory
// this is because vec2s for example are smaller and can change the offset of all
// of our other attributes to be no longer in 16 byte chuncks which the shader cant read in
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    // std::optional<uint32_t> transferFamily; no reason because on m2s all queues are both graphics and transfer, 
    // using a differnt family give no extra performance
    std::optional<uint32_t> presentFamily; // queue to present to a surface

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }

    static int32_t numQueues() {
        return static_cast<int32_t>(sizeof(QueueFamilyIndices) / sizeof(std::optional<uint32_t>));
    }

};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0; // we only have one vertex buffer
        bindingDescription.stride = sizeof(Vertex);
        std::clog << bindingDescription.stride;
        // this says that we want a new struct for each vertex, its per vertex not per instance
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        // this says how to extract the data from the struct, we have two point of data
        // so we need two attribute descriptions
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        // this describes our position data
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        // this is a format for 2 32 bit floats
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        // describing our color data
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; // vec 3 now
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

};
const float size = 1.0f;
const float innerSize = 0.9f;
const float sqrt3 = std::sqrt(3);
const float sqrt3over3 = sqrt3 / 3.0f;
const float sqrt3over6 = sqrt3 / 6.0f;

// const std::vector<Vertex> vertices = {
//     // {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
//     // {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
//     // {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
//     // go in a clockwise direction

//     {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}}, // top left
//     {{1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}}, // bottom right
//     {{-1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}}, // bottom left

//     {{-1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}}, // top left
//     {{1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}}, // top right
//     {{1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}}, // bottom right

//     {{0.0f * size, -sqrt3over3 * size}, {0.0f, 1.0f, 0.0f}},
//     {{0.5f * size, sqrt3over6 * size}, {0.0f, 0.0f, 1.0f}},
//     {{-0.5f * size, sqrt3over6 * size}, {1.0f, 0.0f, 0.0f}},

//     {{0.0f * innerSize, -sqrt3over3 * innerSize}, {1.0f, 0.0f, 0.0f}},
//     {{0.5f * innerSize, sqrt3over6 * innerSize}, {0.0f, 1.0f, 0.0f}},
//     {{-0.5f * innerSize, sqrt3over6 * innerSize}, {0.0f, 0.0f, 1.0f}}
// };

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

// these reuse verticies by just saying the indices of them
const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

// a few proxy function

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
    const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger
) {
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    uint32_t currentFrame = 0;

    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device; // logical device

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkDebugUtilsMessengerEXT debugMessenger;

    VkSurfaceKHR surface;

    VkSwapchainKHR swapChain;

    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews; // defines how to access each image

    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipeline graphicsPipeline;

    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    // vectors because of frames in flight
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;


    bool framebufferResized = false;

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        // https://developer.nvidia.com/vulkan-memory-management
        // combine both buffer into one and use offsets so the memory is closer together
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }
    
    void cleanup() {
        cleanupSwapChain();

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
     
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);


        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);
        
        vkDestroyDevice(device, nullptr);
        
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void initWindow() {
        glfwInit(); // must be called
        // this is here because glfw was made for opengl
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        // make it so its not resizable
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // replace the first nullptr with glfwGetPrimaryMonitor() to go full screen
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this); // mostly because glfw is kinda old and dumb
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    // vk instance

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        // this must match the version we are using of vulkan (i think this will give an error)
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        std::vector<const char*> requiredExtensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();

        // these are both bit fields, |= is bitwise or equals
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

        // this relates to validation layers
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } 
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }


        // generally info is read from a struct, callbacks for memory stuff is passed in
        // and a variable to store a refrence to the result is passed in
        // std::clog << vkCreateInstance(&createInfo, nullptr, &instance) << '\n';
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance");
        }

        uint32_t extensionCount = 0;
        // this will just run to tell us how many extentions there are
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);
        // now we run it again to store all the extensions
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::clog << "available extensions:\n";

        for (const VkExtensionProperties& extension : extensions) {
            std::clog << '\t' << extension.extensionName << '\n';
        }
    }
    
    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            // this fixes a strange mac error
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME); // emplace_back

            // this is for debugging and validation errors
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

            // here to fix error VUID-VkDeviceCreateInfo-pProperties-04451
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }

        return extensions;
    }

    // surface / chain swap

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface");
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        // lots to do with colors and their format
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        // lots to do with how we queue images
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        // lots to do with the pixel dimension of what we are dealing with
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t minImageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (// this is a special case meaning that there is no maximum
            swapChainSupport.capabilities.maxImageCount > 0 &&
            // we want to do this incase the min + 1 exceeds the maxium
            minImageCount > swapChainSupport.capabilities.maxImageCount
        ) {
            minImageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{}; // more create info yay
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;

        createInfo.surface = surface;
        createInfo.minImageCount = minImageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        // this is always one unless unless you are developing a stereoscopic 3D application (doin vr stuff)
        createInfo.imageArrayLayers = 1;
        // this says what types of operations we want to do
        // because we are rendering directly onto an image given by the chain swap this works
        // but if we wanted to prerender an image then use memory transfer operators to
        // move that whole image onto the chain swap then we would need to use 
        // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsFamily.value(), 
            indices.presentFamily.value()
        };

        // images must be transfered from the different queues, 
        // which can be a small problem if they are in different queue families

        // the first option is if they differ, (are in different familes)
        // we use concurrent sharing mode which is less performant
        // but allows really easy transfer of ownership

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } 
        // the other option is exclusive, which is much more performant
        // but hard to manage if you have to transfer ownership of the image
        // between different queue familes (which we dont if this statement runs)
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // optional
            createInfo.pQueueFamilyIndices = nullptr; // optional
        }
        
        // this would be if we wanted to alter the image like flip it or rotate or something
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCompositeAlphaFlagBitsKHR.html
        // ignore the alpha channel
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;
        // this means that the chain swap may not own all of the pixels (other windows stuff like that)
        createInfo.clipped = VK_TRUE;

        // if a window is resized or something the swap chain becomes suboptimal
        // if thats the case then you need to make a new swap chain and this can help
        // doing this can help manage resources and display images that didnt make it

        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain");
        }

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        uint32_t imageCount;
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) { // this means we are minimized
            // make sure that we dont get stuck in this loop and dont close the window when we should
            if (glfwWindowShouldClose(window))
                return;

            // so we just loop
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        // std::clog << "recreating chainswap \n";
        // we wait so we arent touching resources that are in use
        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void cleanupSwapChain() {
        for (VkFramebuffer& framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        
        for (VkImageView& imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        // populate details.capabilities with how our device can interact with our surface
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        
        // get information about the surface formats

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount); // resize to hold all formats
            // populate the vector
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        // very similar with present modes
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const VkSurfaceFormatKHR& availableFormat : availableFormats) {
            if (
                availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && // mostly chosen because we want gamma color space (it is gamma space)
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR // make sure that the format is non linear (gamma space???)
            ) {
                return availableFormat;
            }
        }

        // it cant be empty because of our device checks so if nothing works just use the first one we can use
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        /*
        nice explainer from the tutorial of all the present modes

        VK_PRESENT_MODE_IMMEDIATE_KHR: 
        Images submitted by your application are transferred to the screen right away, which may result in tearing.

        VK_PRESENT_MODE_FIFO_KHR: 
        The swap chain is a queue where the display takes an image from the front of the queue when the display 
        is refreshed and the program inserts rendered images at the back of the queue. If the queue is full then 
        the program has to wait. This is most similar to vertical sync as found in modern games. The moment that 
        the display is refreshed is known as "vertical blank".

        VK_PRESENT_MODE_FIFO_RELAXED_KHR: 
        This mode only differs from the previous one if the application is late and the queue was empty at the last
        vertical blank. Instead of waiting for the next vertical blank, the image is transferred right away when 
        it finally arrives. This may result in visible tearing.

        VK_PRESENT_MODE_MAILBOX_KHR: 
        This is another variation of the second mode. Instead of blocking the application when the queue is full, 
        the images that are already queued are simply replaced with the newer ones. This mode can be used to render 
        frames as fast as possible while still avoiding tearing, resulting in fewer latency issues than standard 
        vertical sync. This is commonly known as "triple buffering", although the existence of three buffers alone 
        does not necessarily mean that the framerate is unlocked.
        */
        
        // mailbox is nice so we use it if we can
        // uses more energy which is a small problem for mobile devices

        for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // if it is the max value that is the surfaces way of saying that we can choose our extent
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            // if we are choosing then we want to find the size of the window
            // we want them to match as close as possible
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            // but we still have to clamp it based on the min and maxes
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    // textures

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        // pass by refrence to allow the function to set them
        stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        // r g b and a value for each pixel
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        // same as before
        createBuffer(
            imageSize, 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, 
            stagingBufferMemory
        );
        // copy data onto this buffer
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
            memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        // now that its copied we can free the memory
        stbi_image_free(pixels);

        // TODO, combine these into a single command buffer
        // from the tutorial:
        /*
            All of the helper functions that submit commands so far have been set up to execute synchronously 
            by waiting for the queue to become idle. For practical applications it is recommended to combine 
            these operations in a single command buffer and execute them asynchronously for higher throughput, 
            especially the transitions and copy in the createTextureImage function. Try to experiment with this 
            by creating a setupCommandBuffer that the helper functions record commands into, and add a 
            flushSetupCommands to execute the commands that have been recorded so far. It's best to do this 
            after the texture mapping works to check if the texture resources are still set up correctly.
        */

        createImage(
            texWidth, texHeight, 
            VK_FORMAT_R8G8B8A8_SRGB, 
            // other option is VK_IMAGE_TILING_LINEAR, which lays out the texels in order in memory
            // but we dont need to access specific values like this so optimal works best for us
            VK_IMAGE_TILING_OPTIMAL, 
            // this image will be a destination after a buffer copy and we want to sample it for our other stuff
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | 
                VK_IMAGE_USAGE_SAMPLED_BIT, 
            // this will now be memory on the gpu so speedy
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            textureImage, textureImageMemory
        );

        // transition it to a format that allows us to most easily populate it from the buffer
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        // transition the image format to be optimal for being read by the shader
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // no longer need the staging stuff
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        // could be 1d or 3d as well 
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        // get our dimensions
        imageInfo.extent.width = static_cast<uint32_t>(width);
        imageInfo.extent.height = static_cast<uint32_t>(height);
        // this is also a dimension, which we are only storing one value for
        imageInfo.extent.depth = 1;
        // random stuff that we arent using for now
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        // this is the same format as how we loaded the image
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        // this just means that the previous image doesnt matter and we can clear it
        // we will likely transition this layout as we use the image for different things
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        // only being used by one queue family
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        // sampling stuff we can ignore
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.flags = 0; // optional

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        // similar to what has been done before
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        // assume theres no offset
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        // same as before
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
            width, // number of values in the dimension
            height,
            1 // one depth value
        };

        // assumes the image layout is VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, which is best for transfering stuff
        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        // barriers are another sync object type thing
        // they are a barrier in the graphics pipeline
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;

        // normally this allows you to transfer ownership of queue families
        // but if you dont want to do this you need to specify
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        barrier.image = image;
        // it is a color image that isnt an array, doesn use mip levels and has a single layer
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        // if we are transitioning from nothing to being able to transfer we dont have much to sync
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            // start at the top
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            // the stage where all the image draws should be ready
            // this is called a pseudo stage, its not a real one
            // docs https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#VkPipelineStageFlagBits
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } 
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            // image needs to be ready for the fragment shader
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } 
        else {
            throw std::invalid_argument("unsupported layout transition");
        }

        vkCmdPipelineBarrier(
            commandBuffer, // our command buffer
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported
            sourceStage, destinationStage, // the stage before our barrier, and the stage that will wait on our barrier
            0, // flags
            0, nullptr, // this is for memory barriers
            0, nullptr, // this is for buffer memory barriers
            1, &barrier // this is for image memory barriers
        );


        endSingleTimeCommands(commandBuffer);
    }

    // image view

    void createTextureImageView() {
        createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        // same subresourceRange setup as anywhere else
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        // viewInfo.components is VK_COMPONENT_SWIZZLE_IDENTITY by default no need to change
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComponentMapping.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComponentSwizzle.html

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view");
        }
        return imageView;
    }

    void createTextureSampler() {
        VkSamplerCreateInfo samplerInfo {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        // this tells us what to do when we over or undersample our image
        // oversampling, meaning we have more fragments then texels, can lead to a blocky look
        // undersampling, meaning the opposite, means that we can loose some higher frequency patterns, which in that case its better to blur it with a linear sample
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        
        // one note is that uvw coordinates indicates that we are in texture space and not world space
        // this address mode just tells us what happens when we sample beyond our image dimensions
        // repeat is the same as just taking the modulo by the image dimensions
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;

        // a way of filtering that deals with textures at sharp angles
        // https://en.wikipedia.org/wiki/Anisotropic_filtering
        // look at the implementation page
        samplerInfo.anisotropyEnable = VK_TRUE;

        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        // this shouldnt change anything because we repeat the image rather then use a clamp to border address mode
        // but if we did sample out of the range we would get an opaque black
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

        // this is telling the sampler that the texels will be on a range from 0 to 1, rather than using the image dimensions
        // this helps us use texture of different sizes
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        // this is if we want to filter out the pixels a certain way, not useful for this application
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

        // todo : explained in a later chapter
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler");
        }
    }

    // graphics pipeline

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        // the type of descriptor that we are using
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        // we only refrence the descriptors from the vertex stage of the pipeline
        uboLayoutBinding.pImmutableSamplers = nullptr; // optional (image sampling stuff)
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout");
        }
    }

    void createGraphicsPipeline() {
        std::vector<char> vertShaderCode = readFile("shaders/spirv/vert.spv");
        std::vector<char> fragShaderCode = readFile("shaders/spirv/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // cuz this is a vert shader
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // the name of the main method

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT; // now its a frag shader
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        // pSpecializationInfo could be used to pass constant into the shader
        // this would allow the compiler to get rid of if statements and can
        // be really fast if we want to configure the shaders at all

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        // these are the things that we want to (and now need to)
        // specify when we draw, almost everything is completely 
        // immutable but if we want to change things dynamically
        // then we specify here, full list of what can be dynamic
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDynamicState.html
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // vertices are hardcoded so we dont need any input (sike not anymore)
        VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = Vertex::getAttributeDescriptions();
        
        // load all the input info from the struct
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        // this means that every 3 verticies are a triangle
        // you could just draw lines, or have verticies be reused etc
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        // do more research on this option please <-------------------------------------------------------------------------------- THIS
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        
        // viewports and scissors are part of the dymanic state
        // we can ignore them here otherwise they become immutable
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // scissor vs viewport extent
        // this is pretty much just stretching (viewport extent)
        // vs cropping (scissor extent)
        // https://vulkan-tutorial.com/images/viewports_scissors.png

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        // if this were true that means that fragments that are beyond the depth
        // range would be clamped to that range instead of having them discarded
        rasterizer.depthClampEnable = VK_FALSE;
        // if this were true then geometry would not pass through
        // this shader and nothing would go to the framebuffer
        rasterizer.rasterizerDiscardEnable = VK_FALSE;

        // other options are to draw only lines, or only points (requires enabling gpu features)
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        // anything else requires gpu features
        rasterizer.lineWidth = 1.0f;
        
        // culling means what do we want to discard
        // we want to discard the back ones because they are behind
        // we could also do no culling (try that out)
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        // just after we draw a triangle from verticies what face is the front
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        // we can alter the depth values in this shader if we want (we dont)
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // optional
        rasterizer.depthBiasClamp = 0.0f; // optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // optional

        // just anti aliasing stuff, its disabled for now
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // optional
        multisampling.pSampleMask = nullptr; // optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // optional
        multisampling.alphaToOneEnable = VK_FALSE; // optional
        
        // color blending is the process of combining the color produced
        // by the fragment shader with the color already present in the
        // frame buffer, this would be helpful for alpha blending (we arent doing that)
        // after its done the color will be & with the write mask
        // so this would just pass everything through and override the previous color
        
        // for implementing alpha blending go back to the tutorial
        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions
        
        // color blending specific to the frame buffer
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = 
            VK_COLOR_COMPONENT_R_BIT | 
            VK_COLOR_COMPONENT_G_BIT | 
            VK_COLOR_COMPONENT_B_BIT | 
            VK_COLOR_COMPONENT_A_BIT;

        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // optional

        // this is global color blending config, it will override anything specified above
        // again, almost everything is disabled
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // optional
        colorBlending.blendConstants[1] = 0.0f; // optional
        colorBlending.blendConstants[2] = 0.0f; // optional
        colorBlending.blendConstants[3] = 0.0f; // optional

        // the pipeline layout is for uniform values and push constants
        // these are both ways of modifying the shaders and passing in info
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        // our descriptor layout
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0; // optional
        pipelineLayoutInfo.pPushConstantRanges = nullptr; // optional

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages; // our shaders

        // this is way too big of a struct
        // but this is all the steps to the graphics pipeline its pretty cool to see
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;

        pipelineInfo.layout = pipelineLayout;

        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0; // this is the index of the subpass

        // these are used if we want to make another graphics pipeline
        // that is very common to this one, we wont use it
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // optional
        pipelineInfo.basePipelineIndex = -1; // optional

        // FINALLY
        // also the VK_NULL_HANDLE is a cache that can help us cache all this info in a file somewhere
        // it can speed up the pipeline creation but we will use it later
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline");
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        // finally a create info stuct that isnt massive
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module");
        }

        return shaderModule;
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        // no multisampling (anti aliasing) so just one sample
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

        // when we load all the data start by clearning it
        // alternatives could be keep the data from before
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // we want to make sure we are storing the image because it will be displayed
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // we arent using the stencil buffer so doesnt matter
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // initial layout doesnt matter because we are clearing it anyways
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // this is the format we want for presenting an image to the swapchain
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        // refers to the color out index (0) in our fragment shader
        colorAttachmentRef.attachment = 0;
        // its a color buffer so optimizing for color is good
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        // we specify that its a graphics subpass (compute subpasses arent supported yet tho)
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{}; // this is all confusing please learn better me
        // this is a place holder for operations that happen outside of the subpass
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        // this is saying that our subpass (index 0) is the one thats dependent
        dependency.dstSubpass = 0;
        // this is saying that source subpass is involved with the color attachement stage
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // this says what type of memory access is involved
        dependency.srcAccessMask = 0;
        // says that our subpass is making use of the color attachement stage
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // and we need to have access to write colors
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass");
        }
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize, 
            // this buffer will be a source in a memory transfer operation
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
            // means that we can get a memory map
            // also means that its pretty slow memory for the gpu
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, 
            stagingBufferMemory
        );

        void* data; // void pointer means that it just points to a spot in memory with no specific data type
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data); // make the memory accessable to the cpu
        // copy all the vertex data to the data pointer we just got
        // so the data pointer has a "virtual address" which is in the cpus memory space
        // but the actual data of it lives on the cpu, so this is kinda like a window
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory); // get rid of this virtual memory space
        // this is all pretty cool i should learn more about it

        createBuffer(
            bufferSize, 
            // this will be a destination in a memory transfer operation as well as a vertex buffer
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
            // we now want the faster local memory
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            vertexBuffer, 
            vertexBufferMemory
        );
        // copy the staging buffer to the very fast vertex buffer on the gpu
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer() {
    // mostly copy paste from the function above so light annotations 
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            stagingBuffer, 
            stagingBufferMemory
        );

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(
            bufferSize, 
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
            indexBuffer, 
            indexBufferMemory
        );
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        // resize
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(
                bufferSize, 
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                    // this way we can have a memory map
                    // having a staging buffer is too slow with how often this is updated
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                uniformBuffers[i], 
                uniformBuffersMemory[i]
            );
            
            // assigns uniformBuffersMapped to a virtual memory adress poiting to a spot on the gpu (our uniformBuffersMemory)
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }

    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        // we have as many uniform buffers as we do frames in flight
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<u_int32_t>(poolSizes.size());
        // its very strange how this has its own struct it just represents ints
        poolInfo.pPoolSizes = poolSizes.data();
        // this is a max value across all pools
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        // if we want to be able to free the buffers induvisually
        // use VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        poolInfo.flags = 0;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool");
        }
    }
    
    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        // combine the descriptor pools and the layouts
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // info about the buffers we made
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            // descriptors can be arrays so we just say start at index 0
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            // we choose one of the 3, we are are using buffer data
            descriptorWrites[0].pBufferInfo = &bufferInfo;
            descriptorWrites[0].pImageInfo = nullptr; // optional
            descriptorWrites[0].pTexelBufferView = nullptr; // optional

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(device, static_cast<u_int32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }

    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory");
        }
        // this is pretty cool
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // we are only running one command on the buffer
        // letting the driver know can help it optimize
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        // free the buffer from memory
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // optional
        copyRegion.dstOffset = 0; // optional
        copyRegion.size = size;
        // we are only moving one region around, no offsets
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        // weve recorded all that we need
        endSingleTimeCommands(commandBuffer);
        
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        // << is also a bit shift operator
        // so what this does is goes through all the memory types
        // and for each one checks if a flag bit associated with it is flipped
        // typeFilter is a bitfield of suitable memory types
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
            // this checks the properties to see if all the properties are met
                ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            ) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type");
    }

    // frame buffers
    
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };
            
            // its very similar to the image view but you specify the render pass
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1; // again we arent using any fancy double image stuff

            // pass in a refrence to an index of our member varible to store it in
            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer");
            }
        }

    }

    // commands

    void createCommandPool() {
        // command pools are memory managers for command buffers
        // command buffers store commands that make up us telling the gpu what to do
        // and it happens many times so memory management for this is important
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // this allows us to reset each command buffer individually, which we want
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        // buffers are executed by sending a command to a queue so we want a refrence here
        // any single pool can only exicute commands on a single queue
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool");
        }
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool; // for memory management
        // secondary command buffers dont directly submit stuff to a queue
        // but can be called by primary buffers, useful for reused stuff
        // pretty much used for chucks of reused commands for performance
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // optional
        beginInfo.pInheritanceInfo = nullptr; // optional (only used for secondary buffers)

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

        // defines what area of the image we want to run the render pass on
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColor = {{{0.02f, 0.02f, 0.02f, 1.0f}}}; // change this value around i think it will effect how the background appears
        // just tried this and it does thats pretty cool
        renderPassInfo.clearValueCount = 1; // other clear values would be like clearing depth etc
        renderPassInfo.pClearValues = &clearColor;

        // we specify that we arent using secondary buffers with the last arg
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        // second arg says that this is a graphics, not a compute pipeline
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // our viewport and scissor was set to be dynamic so we need to specify them
        // all pretty self explaintory stuff
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        // descriptors
        vkCmdBindDescriptorSets(
            commandBuffer, 
            // could be on a compute pipeline so we specify
            VK_PIPELINE_BIND_POINT_GRAPHICS, 
            pipelineLayout, 0, 1, 
            &descriptorSets[currentFrame], 
            0, 
            nullptr
        );

        // after 1150 lines of code, we draw a triangle
        // third param is how many instances we want to draw
        // last two params are offsets for the counts
        // haha jokes on you its no longer hard coded and now its 1500 lines (update 2000)
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);


        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer");
        }
    }

    // logical device

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;

            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{}; // dont need specific features atm (not anymore haha)
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());


        createInfo.pEnabledFeatures = &deviceFeatures;

        // the following is similar to creating a vkinstance
        // but now its all specific to devices

        // dont need any driver specific extensions
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // this is not needed, validation layers the same across a vkinstance
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device");
        }

        // now use the device and the queues we want (only one) to get
        // a refrence and store it inside graphicsQueue
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

    }

    // physical device

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("there are no gpus with vulkan support");
        }

        // make a list for the devices and then enumerate again to populate it
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        VkPhysicalDevice t;


        for (const VkPhysicalDevice& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                // break;
            }
        }

        // if the device wasnt set then none are suitable
        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable gpu with vulkan support");
        }

        // this just picks the first suitable gpu
        // another option is to gather information and rank the gpus
        // might favor dedicated rather than integrated, etc
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            // make sure that there are display formats and present modes
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        // check all the queues on the gpu
        if (indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy) {
            std::clog << "GPU found "  << deviceProperties.deviceName << '\n';
            return true;
        }
        return false;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        // kinda like ticking a check box, it goes through and tries to erase all the requirements
        for (const VkExtensionProperties& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        // if it didnt meet all of the requirements then return false
        return requiredExtensions.empty();

    }

    // queue

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
            // break if weve already made the checks we need
            if (indices.isComplete()) {
                break;
            }

            // check for presentation support
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            // checks to see if the queue graphics flag is present
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }
            
            i++;
        }
        
        return indices;
    } 
    
    // sync objects

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        // semaphores are a way of linking the exicution of one queue operation to the start of another
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        // fences pause operation on the cpu completely, dont use them unless there is a reason
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // the fence will start unsignaled, and only becomes signaled after a frame is drawn
        // this means that there is no way for the first operation to start because its waiting
        // on the (non existant) one before it to finish exicution
        // therefore we start the fence off in its signaled state so it will get started
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS
            ) {
                throw std::runtime_error("failed to create sync objects");
            }
        }
    }

    // validation layer (debugging) stuff

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

        createInfo.messageSeverity = // leaves out informational messages
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        createInfo.messageType = // everything
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // optional
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger");
        }
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: Diagnostic message
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: Informational message like the creation of a resource
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: Message about behavior that is not necessarily an error, but very likely a bug in your application
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: Message about behavior that is invalid and may cause crashes
        VkDebugUtilsMessageTypeFlagsEXT messageType, // VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: Some event has happened that is unrelated to the specification or performance
        // VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT: Something has happened that violates the specification or indicates a possible mistake
        // VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: Potential non-optimal use of Vulkan
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, 
        void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << '\n' << std::endl;

        return VK_FALSE;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;

        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // now check to see if the layers we specify exist in the avaible layers
        // iterate through the validation layers we specify
        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            // iterates through the avaible layers
            for (const VkLayerProperties& layerProperties : availableLayers) {
                // checks if the layers are equal
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }
        return true;
    }

    // random helpers

    static std::vector<char> readFile(const std::string& fileName) {
        // ill read about this more later but the second arg seems to
        // act kinda like a bit field (i say that because of the bitwise
        // operator and the meaning of the ate and binary)
        // ate means we read starting from the end of the file
        // binary means we dont try to turn it into text, just 
        // read it as binary
        std::ifstream file(fileName, std::ios::ate | std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file " + fileName);
        }
        // this takes the current positon of where we are in the
        // file stream (tellg) and because we read it backwards
        // that position is the same as the file size
        size_t fileSize = (size_t) file.tellg();
        // this will create a buffer for everything in the file
        std::vector<char> buffer(fileSize);
        // now move us back to the start
        file.seekg(0);
        // and read the whole file, store it in the buffer
        file.read(buffer.data(), fileSize);
        file.close();
        
        return buffer;
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // we want to wait so our sync objects dont get mad :(
        // validation error will be upset if we end while semaphores or fences are destroyed while being used
        vkDeviceWaitIdle(device);
    }
      
    void drawFrame() {
        // wait on the signaled fence (our cpu is now idle)
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        // third arg is a timeout, we dont want that so we wait as long as possible
        // approx 6 hundred billion years fyi
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        // possible errors are VK_ERROR_OUT_OF_DATE_KHR, meaning that we cant do anything the surface is incompatable
        // and VK_SUBOPTIMAL_KHR means that we can still get images from the chainswap, but presenting wont work

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image");
        }

        // swap the fence back to unsignaled
        // do this after checking the chainswap to prevent deadlock
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        //
        updateUniformBuffer(currentFrame);

        // empty our command buffer
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        // load all the commands we need onto this buffer
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]}; // we wait until the image is avaible
        VkPipelineStageFlags waitStages[] = {
            // this is the stage where all the final colors have been written
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };

        submitInfo.waitSemaphoreCount = 1; // only waiting on one semaphamore
        // the index of each semaphore corrisponds to the index of each wait stage
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        // this will turn our render finished semaphore into a signaled semaphore when its done
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        // wait until we are done rendering
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        // omg finally my headphones litteraly just died im so tired
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            // now the suboptimal error is a problem so we need to recreate it
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image");
        }

        currentFrame += 1; // increment
        currentFrame %= MAX_FRAMES_IN_FLIGHT; // cap
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static std::__1::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();
        // static so its defined once and stays at that

        std::__1::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
        // these chrono data types are insane
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};


        ubo.model = glm::translate(glm::mat4(1.0f), glm::vec3(
            glm::cos(time/2 * glm::pi<float>()),
            glm::sin(time/2 * glm::pi<float>()), 
            0
        ));
        // (left right, front back, up down)

        // our model will be rotating in world space
        // model space -> world space
        // this is quaternion stuff ill break down the source code later
        ubo.model *= glm::rotate(
            glm::mat4(1.0f), // take an identity matrix
            time * glm::radians(90.0f), // 90 deg a second
            glm::vec3(0.0f, 0.0f, 1.0f) // around the z axis
        );


        // std::clog << "translate " << glm::to_string(glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0))) << "\nrotation " << glm::to_string(ubo.model) << "\nid*rot " << glm::to_string(ubo.model * glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0))) << "\n";


        // world space -> camera space
        ubo.view = glm::lookAt(
            glm::vec3(1.0f, 1.0f, 3.0f), // look from / eye
            glm::vec3(0.0f, 0.0f, 0.0f), // look at / taget
            glm::vec3(0.0f, 0.0f, 1.0f) // maybe should be 0 1 0 ?? but this is the up vector (future: no it isnt)
        );

        /*
        explination of the look at matrix:
        // this is our forward vector, it points right from the center to the eye
        ec<3, T, Q> const f(normalize(center - eye));

        // this vector is perpendicular to both our  forward and up vector
        // following the right hand rule this new vector points right relative to the camera
		vec<3, T, Q> const s(normalize(cross(f, up)));

        // this new vector now points up relative to the camera
		vec<3, T, Q> const u(cross(s, f));
        
        // all these are unit vectors, we dont care about the other properties of the cross product

		mat<4, 4, T, Q> Result(1);
        // these are pretty much our basis vectors
        // for this transformation now
		Result[0][0] = s.x;
		Result[1][0] = s.y;
		Result[2][0] = s.z;

		Result[0][1] = u.x;
		Result[1][1] = u.y;
		Result[2][1] = u.z;

        // this is flipped because generally the camera looks down the negative z axis
		Result[0][2] = -f.x;
		Result[1][2] = -f.y;
		Result[2][2] = -f.z;

        // our coordinates are homogenous
        // so this represents a translation
        // these project the eye (camera) onto its own coordinate system
        // they are negative (have their signs flipped in the case of f) 
        // because as we move the camera forward (or any dir), 
        // we want everything else to move backwards (opposite dir)
		Result[3][0] = -dot(s, eye);
		Result[3][1] = -dot(u, eye);
		Result[3][2] = dot(f, eye);
		return Result;
        */

        ubo.proj = glm::perspective(
            glm::radians(90.0f),
            swapChainExtent.width / (float) swapChainExtent.height, 
            0.1f, // near plane 
            10.0f // far plane
        );

        /*
        now an explination for this source code since its abstracted
        // some random assertions to prevent negatives
        assert(width > static_cast<T>(0));
		assert(height > static_cast<T>(0));
		assert(fov > static_cast<T>(0));


		T const rad = fov;

        // get the tanget of the fov, pretty much the ratio of the height of the viewport to the focal length
        // this seems to be assuming a focal length of one (maybe something to change in the future???)
        // in that case its just the height of the viewport
		T const h = glm::cos(static_cast<T>(0.5) * rad) / glm::sin(static_cast<T>(0.5) * rad);
        // width of the viewport very simple
		T const w = h * height / width; ///todo max(width , Height) / min(width , Height)?

        // casting 0 is pretty funny idk why
		mat<4, 4, T, defaultp> Result(static_cast<T>(0));
        
		Result[0][0] = w; // our x coord is scaled by our width
		Result[1][1] = h; // our y coord is scaled by our height
        // the projection comes later with the x coordinate

        // this took me so long to understand but omg this is SO COOL
        // so what this does is make it so the w coord (almost always 1) in the output is
        // equal to the -z of the camera, or the depth (its negative because thats flipped in camera space)
        // so now what this does it is scales everything by 1/z, or 1 / the depth, making it so
        // that the futher things are, the closer to the center they are
        // THATS HOW PERSPECTIVE WORKS FOR HOMOGENOUS COORDINATES
		Result[2][3] = - static_cast<T>(1);

        // these are terms that scale then translate the depth to be on a very specific range before it goes through for futher processing
        // so i finally figured this out, i had a problem figuring out the terms because i forgot forward in camera space was the -z dir
        // so we want to map all of our depth coordinates from zNear to zFar on to the range -1, to 1.
        // because depth is flipped we come up with the conditions that f(-zNear) = -1 and f(-zFar) = 1
        // again because of perspective we want this to be the depth coordinates in ndc (normalized device coordinates, our target space)
        // to be inversely related to our camera space depth coordinates, so it will be generally modeld by the function 1/x
        // adding terms to find this function we can get something like f(x) = (b/x) + a, then rearrange to get  f(x) = (ax + b) / x
        // notice x in the denominator, when we solve for this function after the matrix all terms will be multipled by w^-1, which will be -zcamera
        // and since our function here takes in -zcamera that is equivalent to our x in the denominator
        // now how we scale x is just with the (2,2) matrix term, just like normal, so that is our a term in the function
        // now how we translate x is with the (3,2) term, thats because with homogenous coordinates we guarante (in this context)
        // that our w part of the vector is 1, so then scaling it by a certain amount will alway add a consistent number to the matrix
        // so (3,2) will be our b term
        
        // now just solve for the function for a and b in f(x) = (ax + b) / x given the conditions that f(-zNear) = -1 and f(-zFar) = 1
        // you should get terms that match up with the glm matrix
        // here is my work in desmos https://www.desmos.com/calculator/2zbxhbzwm4

		Result[2][2] = - (zFar + zNear) / (zFar - zNear);
		Result[3][2] = - (static_cast<T>(2) * zFar * zNear) / (zFar - zNear);
		return Result;
        */

        // this is the number that determines the y value of the
        // basis y vector, so flipping it flips y value of everything going through the matrix
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
